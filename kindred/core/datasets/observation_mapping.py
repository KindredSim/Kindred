"""Pure observation-column mapping and exact-zero seeding helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


class ObservationMappingError(ValueError):
    """Raised when an explicit observation mapping cannot be honored."""


@dataclass(frozen=True)
class ObservationMappingSuggestion:
    raw_column: str
    model_species: str


def normalize_observation_mapping(
    mapping: Iterable[tuple[str, str]],
    *,
    raw_columns: Sequence[str],
    model_species: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Validate and deterministically order an explicit partial bijection."""
    normalized_columns = tuple(str(value).strip() for value in raw_columns)
    normalized_species = tuple(str(value).strip() for value in model_species)
    if any(not value for value in normalized_columns):
        raise ObservationMappingError("Raw observation column names must be nonblank.")
    if any(not value for value in normalized_species):
        raise ObservationMappingError("Model species names must be nonblank.")
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ObservationMappingError("Raw observation column names must be unique.")
    if len(set(normalized_species)) != len(normalized_species):
        raise ObservationMappingError("Model species names must be unique.")
    available_columns = set(normalized_columns)
    available_species = set(normalized_species)
    by_column: dict[str, str] = {}
    by_species: dict[str, str] = {}
    for raw_column, raw_species in mapping:
        column = str(raw_column).strip()
        species = str(raw_species).strip()
        if not column or not species:
            raise ObservationMappingError("Observation mappings require nonblank column and species names.")
        if column not in available_columns:
            raise ObservationMappingError(
                f"Explicit observation column '{column}' is no longer present in the raw dataset."
            )
        if species not in available_species:
            raise ObservationMappingError(
                f"Explicit observation target '{species}' is not present in the active mechanism."
            )
        previous_species = by_column.get(column)
        if previous_species is not None:
            if previous_species == species:
                raise ObservationMappingError(
                    f"Observation mapping '{column}' to '{species}' is duplicated."
                )
            raise ObservationMappingError(
                f"Observation column '{column}' is mapped to more than one mechanism species."
            )
        previous_column = by_species.get(species)
        if previous_column is not None:
            raise ObservationMappingError(
                f"Mechanism species '{species}' is mapped from more than one observation column."
            )
        by_column[column] = species
        by_species[species] = column
    return tuple(
        (column, by_column[column])
        for column in normalized_columns
        if column in by_column
    )


def suggest_observation_mapping(
    *,
    raw_columns: Sequence[str],
    model_species: Sequence[str],
) -> tuple[ObservationMappingSuggestion, ...]:
    """Suggest only unambiguous exact-name matches.

    Suggestions are never a confirmed mapping. Callers must present and obtain
    explicit confirmation before passing them to a canonical mutation.
    """
    columns = tuple(str(value) for value in raw_columns)
    suggestions: list[ObservationMappingSuggestion] = []
    claimed_columns: set[str] = set()
    claimed_species: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for species_value in model_species:
        species = str(species_value)
        exact = [column for column in columns if column == species]
        if len(exact) == 1:
            candidates.append((exact[0], species))
    for column, species in candidates:
        if column in claimed_columns or species in claimed_species:
            continue
        claimed_columns.add(column)
        claimed_species.add(species)
        suggestions.append(
            ObservationMappingSuggestion(
                raw_column=column,
                model_species=species,
            )
        )
    return tuple(suggestions)


def seed_species_values_at_exact_zero(
    *,
    observations: Mapping[str, object],
    model_species: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
) -> tuple[tuple[str, float], ...]:
    """Build a complete species vector using only confirmed samples at t == 0.0."""
    columns = tuple(str(value) for value in observations.keys())
    mapping = normalize_observation_mapping(
        confirmed_mapping,
        raw_columns=columns,
        model_species=model_species,
    )
    column_by_species = {species: column for column, species in mapping}
    seeded: list[tuple[str, float]] = []
    for species_value in model_species:
        species = str(species_value)
        value = 0.0
        column = column_by_species.get(species)
        if column is not None:
            spec = observations[column]
            if not isinstance(spec, Mapping):
                raise ObservationMappingError(
                    f"Observation column '{column}' has an invalid series payload."
                )
            t_values = tuple(float(item) for item in spec.get("t", ()))
            y_values = tuple(float(item) for item in spec.get("y", ()))
            if len(t_values) != len(y_values):
                raise ObservationMappingError(
                    f"Observation column '{column}' has mismatched time and value lengths."
                )
            source_rows = tuple(int(item) for item in spec.get("source_rows", ()))
            if (
                len(source_rows) != len(t_values)
                or any(row <= 0 for row in source_rows)
                or len(set(source_rows)) != len(source_rows)
            ):
                raise ObservationMappingError(
                    f"Observation column '{column}' requires one unique positive "
                    "physical source row per value."
                )
            zero_indexes = tuple(
                index
                for index, t_value in enumerate(t_values)
                if t_value == 0.0
            )
            if len(zero_indexes) > 1:
                identities = tuple(
                    source_rows[index] for index in zero_indexes
                )
                joined = ", ".join(str(identity) for identity in identities)
                raise ObservationMappingError(
                    f"Observation column '{column}' mapped to species '{species}' "
                    f"has multiple exact t=0.0 values at source rows {joined}."
                )
            if zero_indexes:
                y_value = y_values[zero_indexes[0]]
                if not math.isfinite(y_value):
                    raise ObservationMappingError(
                        f"Observation column '{column}' has a non-finite value at t=0."
                    )
                value = y_value
        seeded.append((species, value))
    return tuple(seeded)
