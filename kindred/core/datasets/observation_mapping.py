"""Pure raw-observation mapping and exact-zero seeding helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from kindred.core.species import validate_imported_time_seconds, validate_observation


class ObservationMappingError(ValueError):
    """Raised when an explicit observation mapping cannot be honored."""


@dataclass(frozen=True)
class ObservationMappingSuggestion:
    raw_observation_id: str
    model_target: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.raw_observation_id, "raw observation ID"),
            (self.model_target, "model target"),
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
    model_targets: Sequence[str],
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
    normalized_targets = tuple(
        _canonical_text(value, "Model target name")
        for value in model_targets
    )
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ObservationMappingError("Raw observation IDs must be unique.")
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ObservationMappingError("Model target names must be unique.")
    available_ids = set(normalized_ids)
    column_by_id = dict(normalized_observations)
    available_targets = set(normalized_targets)
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
    observations = tuple(
        (
            _canonical_text(raw_id, "Raw observation ID"),
            _canonical_text(raw_column, "Raw observation label"),
        )
        for raw_id, raw_column in raw_observations
    )
    suggestions: list[ObservationMappingSuggestion] = []
    claimed_ids: set[str] = set()
    claimed_targets: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for target_value in model_targets:
        target = _canonical_text(target_value, "Model target name")
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


def seed_species_values_at_exact_zero(
    *,
    observations: Mapping[str, object],
    mechanism_species: Sequence[str],
    model_targets: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
) -> tuple[tuple[str, float], ...]:
    """Seed mechanism species from confirmed samples at exact t == 0.0.

    The model-target roster validates every confirmed mapping, including
    authored observables, while only mechanism species receive IC values.
    """
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
        model_targets=model_targets,
    )
    raw_id_by_target = {target: raw_id for raw_id, target in mapping}
    normalized_mechanism_species = tuple(
        _canonical_text(value, "Mechanism species name")
        for value in mechanism_species
    )
    if len(set(normalized_mechanism_species)) != len(
        normalized_mechanism_species
    ):
        raise ObservationMappingError(
            "Mechanism species names must be unique."
        )
    seeded: list[tuple[str, float]] = []
    for species in normalized_mechanism_species:
        value = 0.0
        raw_id = raw_id_by_target.get(species)
        if raw_id is not None:
            spec = observations[raw_id]
            raw_column = _canonical_text(
                spec.get("raw_column"),
                "Raw observation label",
            )
            try:
                t_values = tuple(
                    validate_imported_time_seconds(item, field="observation time")
                    for item in spec.get("t", ())
                )
                y_values = tuple(
                    validate_observation(item, field="observation value")
                    for item in spec.get("y", ())
                )
            except (TypeError, ValueError) as exc:
                raise ObservationMappingError(str(exc)) from exc
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
