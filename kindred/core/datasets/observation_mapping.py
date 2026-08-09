"""Pure raw-observation mapping and exact-zero seeding helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


class ObservationMappingError(ValueError):
    """Raised when an explicit observation mapping cannot be honored."""


@dataclass(frozen=True)
class ObservationMappingSuggestion:
    raw_observation_id: str
    model_species: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.raw_observation_id, "raw observation ID"),
            (self.model_species, "model species"),
        ):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"Observation mapping suggestion {label} must be "
                    "nonblank and edge-normalized."
                )


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
    model_species: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Validate and deterministically order an explicit partial bijection."""
    normalized_observations = tuple(
        (
            _canonical_text(raw_id, "Raw observation ID"),
            _canonical_text(raw_column, "Raw observation label"),
        )
        for raw_id, raw_column in raw_observations
    )
    normalized_ids = tuple(raw_id for raw_id, _ in normalized_observations)
    normalized_species = tuple(
        _canonical_text(value, "Model species name")
        for value in model_species
    )
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ObservationMappingError("Raw observation IDs must be unique.")
    if len(set(normalized_species)) != len(normalized_species):
        raise ObservationMappingError("Model species names must be unique.")
    available_ids = set(normalized_ids)
    column_by_id = dict(normalized_observations)
    available_species = set(normalized_species)
    by_raw_id: dict[str, str] = {}
    raw_id_by_species: dict[str, str] = {}
    for raw_observation_id, raw_species in mapping:
        raw_id = _canonical_text(
            raw_observation_id,
            "Mapped raw observation ID",
        )
        species = _canonical_text(raw_species, "Mapped model species")
        if raw_id not in available_ids:
            raise ObservationMappingError(
                f"Explicit raw observation ID {raw_id!r} is no longer present "
                "in the dataset."
            )
        if species not in available_species:
            raise ObservationMappingError(
                f"Explicit observation target '{species}' is not present in the active mechanism."
            )
        previous_species = by_raw_id.get(raw_id)
        if previous_species is not None:
            if previous_species == species:
                raise ObservationMappingError(
                    f"Observation mapping for raw ID {raw_id!r} "
                    f"({column_by_id[raw_id]!r}) to {species!r} is duplicated."
                )
            raise ObservationMappingError(
                f"Raw observation ID {raw_id!r} ({column_by_id[raw_id]!r}) "
                "is mapped to more than one mechanism species."
            )
        previous_raw_id = raw_id_by_species.get(species)
        if previous_raw_id is not None:
            raise ObservationMappingError(
                f"Mechanism species {species!r} is mapped from more than one "
                "raw observation."
            )
        by_raw_id[raw_id] = species
        raw_id_by_species[species] = raw_id
    return tuple(
        (raw_id, by_raw_id[raw_id])
        for raw_id in normalized_ids
        if raw_id in by_raw_id
    )


def suggest_observation_mapping(
    *,
    raw_observations: Sequence[tuple[str, str]],
    model_species: Sequence[str],
) -> tuple[ObservationMappingSuggestion, ...]:
    """Suggest only unambiguous exact-name matches.

    Suggestions are never a confirmed mapping. Callers must present and obtain
    explicit confirmation before passing them to a canonical mutation.
    """
    observations = tuple(
        (
            _canonical_text(raw_id, "Raw observation ID"),
            _canonical_text(raw_column, "Raw observation label"),
        )
        for raw_id, raw_column in raw_observations
    )
    suggestions: list[ObservationMappingSuggestion] = []
    claimed_ids: set[str] = set()
    claimed_species: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for species_value in model_species:
        species = _canonical_text(species_value, "Model species name")
        exact = [
            (raw_id, raw_column)
            for raw_id, raw_column in observations
            if raw_column == species
        ]
        if len(exact) == 1:
            candidates.append((exact[0][0], species))
    for raw_id, species in candidates:
        if raw_id in claimed_ids or species in claimed_species:
            continue
        claimed_ids.add(raw_id)
        claimed_species.add(species)
        suggestions.append(
            ObservationMappingSuggestion(
                raw_observation_id=raw_id,
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
    raw_observations: list[tuple[str, str]] = []
    for raw_id_value, raw_spec in observations.items():
        raw_id = _canonical_text(raw_id_value, "Raw observation ID")
        if not isinstance(raw_spec, Mapping):
            raise ObservationMappingError(
                f"Raw observation {raw_id!r} has an invalid series payload."
            )
        raw_observations.append(
            (
                raw_id,
                _canonical_text(
                    raw_spec.get("raw_column"),
                    "Raw observation label",
                ),
            )
        )
    mapping = normalize_observation_mapping(
        confirmed_mapping,
        raw_observations=tuple(raw_observations),
        model_species=model_species,
    )
    raw_id_by_species = {species: raw_id for raw_id, species in mapping}
    seeded: list[tuple[str, float]] = []
    for species_value in model_species:
        species = _canonical_text(species_value, "Model species name")
        value = 0.0
        raw_id = raw_id_by_species.get(species)
        if raw_id is not None:
            spec = observations[raw_id]
            raw_column = _canonical_text(
                spec.get("raw_column"),
                "Raw observation label",
            )
            t_values = tuple(float(item) for item in spec.get("t", ()))
            y_values = tuple(float(item) for item in spec.get("y", ()))
            if len(t_values) != len(y_values):
                raise ObservationMappingError(
                    f"Observation {raw_column!r} (raw ID {raw_id!r}) has "
                    "mismatched time and value lengths."
                )
            source_rows = tuple(int(item) for item in spec.get("source_rows", ()))
            if (
                len(source_rows) != len(t_values)
                or any(row <= 0 for row in source_rows)
                or len(set(source_rows)) != len(source_rows)
            ):
                raise ObservationMappingError(
                    f"Observation {raw_column!r} (raw ID {raw_id!r}) requires "
                    "one unique positive physical source row per value."
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
                    f"Observation {raw_column!r} (raw ID {raw_id!r}) mapped "
                    f"to species {species!r} has multiple exact t=0.0 values "
                    f"at source rows {joined}."
                )
            if zero_indexes:
                y_value = y_values[zero_indexes[0]]
                if not math.isfinite(y_value):
                    raise ObservationMappingError(
                        f"Observation {raw_column!r} (raw ID {raw_id!r}) "
                        "has a non-finite value at t=0."
                    )
                value = y_value
        seeded.append((species, value))
    return tuple(seeded)
