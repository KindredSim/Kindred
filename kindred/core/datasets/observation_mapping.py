"""Pure observation mapping and explicit initial-concentration extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from kindred.core.datasets.foundation import ObservationSeries
from kindred.core.quantity_units import MOLAR
from kindred.core.species import validate_concentration


class ObservationMappingError(ValueError):
    """Raised when an explicit observation mapping cannot be honored."""


class InitialConcentrationSource(Enum):
    RECORDED_ZERO = "recorded_zero"
    FIRST_DATA_ROW = "first_data_row"


@dataclass(frozen=True)
class ObservationMappingSuggestion:
    raw_observation_id: str
    model_target: str


def _canonical_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ObservationMappingError(
            f"{label} must be a nonblank edge-normalized string."
        )
    return value


def normalize_observation_mapping(
    mapping: Iterable[tuple[str, str]],
    *,
    raw_observations: Sequence[tuple[str, str]],
    model_targets: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Validate and deterministically order an explicit partial bijection."""
    # Raw observations come from admitted records, targets from the parsed mechanism.
    normalized_ids = tuple(raw_id for raw_id, _ in raw_observations)
    available_ids = set(normalized_ids)
    column_by_id = dict(raw_observations)
    available_targets = set(model_targets)
    by_raw_id: dict[str, str] = {}
    raw_id_by_target: dict[str, str] = {}
    for raw_observation_id, raw_target in mapping:
        raw_id = _canonical_text(
            raw_observation_id,
            "Mapped raw observation ID",
        )
        target = _canonical_text(raw_target, "Mapped model target")
        if raw_id not in available_ids:
            raise ObservationMappingError(
                f"Explicit raw observation ID {raw_id!r} is no longer present "
                "in the dataset."
            )
        if target not in available_targets:
            raise ObservationMappingError(
                f"Explicit observation target '{target}' is not present in the "
                "accepted model target roster."
            )
        previous_target = by_raw_id.get(raw_id)
        if previous_target is not None:
            if previous_target == target:
                raise ObservationMappingError(
                    f"Observation mapping for raw ID {raw_id!r} "
                    f"({column_by_id[raw_id]!r}) to {target!r} is duplicated."
                )
            raise ObservationMappingError(
                f"Raw observation ID {raw_id!r} ({column_by_id[raw_id]!r}) "
                "is mapped to more than one model target."
            )
        previous_raw_id = raw_id_by_target.get(target)
        if previous_raw_id is not None:
            raise ObservationMappingError(
                f"Model target {target!r} is mapped from more than one "
                "raw observation."
            )
        by_raw_id[raw_id] = target
        raw_id_by_target[target] = raw_id
    return tuple(
        (raw_id, by_raw_id[raw_id])
        for raw_id in normalized_ids
        if raw_id in by_raw_id
    )


def suggest_observation_mapping(
    *,
    raw_observations: Sequence[tuple[str, str]],
    model_targets: Sequence[str],
) -> tuple[ObservationMappingSuggestion, ...]:
    """Suggest only unambiguous exact-name matches.

    Suggestions are never a confirmed mapping. Callers must present and obtain
    explicit confirmation before passing them to a canonical mutation.
    """
    observations = tuple(raw_observations)
    suggestions: list[ObservationMappingSuggestion] = []
    claimed_ids: set[str] = set()
    claimed_targets: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for target in model_targets:
        exact = [
            (raw_id, raw_column)
            for raw_id, raw_column in observations
            if raw_column == target
        ]
        if len(exact) == 1:
            candidates.append((exact[0][0], target))
    for raw_id, target in candidates:
        if raw_id in claimed_ids or target in claimed_targets:
            continue
        claimed_ids.add(raw_id)
        claimed_targets.add(target)
        suggestions.append(
            ObservationMappingSuggestion(
                raw_observation_id=raw_id,
                model_target=target,
            )
        )
    return tuple(suggestions)


def supplies_concentration(series: ObservationSeries) -> bool:
    """Whether a measured column can supply initial concentrations."""
    return series.unit is not None and series.unit.compatible(MOLAR)


def first_data_row(observations: Sequence[ObservationSeries]) -> int:
    """The one common dataset row that First data row reads."""
    return min(row for series in observations for row in series.source_rows)


def initial_samples(
    series: ObservationSeries,
    *,
    source: InitialConcentrationSource,
    first_row: int,
) -> tuple[tuple[int, float], ...]:
    """Every (dataset row, value) of one column at the chosen start."""
    if source is InitialConcentrationSource.RECORDED_ZERO:
        coordinates, selected = series.t, 0.0
    else:
        coordinates, selected = series.source_rows, first_row
    return tuple(
        (series.source_rows[index], series.y[index])
        for index, item in enumerate(coordinates)
        if item == selected
    )


def admit_measured_start(value: float, *, species: str) -> float:
    """Admit one measured starting concentration for a species."""
    # A measured start at or below zero (baseline noise, or -0) supplies
    # exact zero; a concentration cannot be negative.
    measured = float(value)
    try:
        return validate_concentration(
            0.0 if measured <= 0.0 else measured,
            field=f"Initial concentration for {species}",
        )
    except (TypeError, ValueError) as exc:
        raise ObservationMappingError(str(exc)) from exc


def extract_initial_concentrations(
    *,
    observations: Sequence[ObservationSeries],
    mechanism_species: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
    source: InitialConcentrationSource,
) -> tuple[tuple[str, float | None], ...]:
    """Read admitted observations through a confirmed mapping for IC creation.

    None means no concentration was supplied; a measured zero remains zero.
    Values already use scientific units and undergo only IC-domain admission.
    """
    if source is InitialConcentrationSource.RECORDED_ZERO and not any(
        0.0 in series.t for series in observations
    ):
        raise ObservationMappingError("The dataset has no recorded observation at t = 0.")
    first_row = first_data_row(observations)
    by_id = {series.raw_observation_id: series for series in observations}
    raw_id_by_target = {target: raw_id for raw_id, target in confirmed_mapping}
    concentrations: list[tuple[str, float | None]] = []
    for species in mechanism_species:
        value = None
        raw_id = raw_id_by_target.get(species)
        if raw_id is not None and supplies_concentration(by_id[raw_id]):
            series = by_id[raw_id]
            samples = initial_samples(series, source=source, first_row=first_row)
            if len(samples) > 1:
                rows = ", ".join(str(row) for row, _value in samples)
                raise ObservationMappingError(
                    f"Column {series.column_name!r} has {len(samples)} measurements at "
                    f"t = 0 (rows {rows}); Recorded data at t = 0 needs exactly one."
                )
            if samples:
                value = admit_measured_start(samples[0][1], species=species)
        concentrations.append((species, value))
    return tuple(concentrations)
