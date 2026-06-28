from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kindred.core.batch_initial_conditions import (
    _strip_named_initial_concentration_sets,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

__all__ = ["MechanismSessionOwner", "ValidationResult"]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error_message: str
    species_count: int
    reaction_count: int
    equilibria_count: int


class MechanismSessionOwner:
    def __init__(self, topology_validator: Callable[[], bool] | None = None) -> None:
        self._canonical_source = MechanismAuthoringSource()
        self._topology_validator = topology_validator
        self._canonical_validation: ValidationResult | None = None

    @property
    def canonical_source(self) -> MechanismAuthoringSource:
        return self._canonical_source

    @property
    def canonical_reactions_text(self) -> str:
        return self._canonical_source.reactions_text

    @property
    def canonical_state_network_dsl(self) -> str:
        return self._canonical_source.state_network_dsl

    @property
    def canonical_full_dsl(self) -> str:
        return self._canonical_source.full_dsl

    def apply_authoritative_source(self, source: MechanismAuthoringSource) -> None:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")
        self._canonical_source = source
        self._canonical_validation = None

    def source_for_reactions_text(self, text: str) -> MechanismAuthoringSource:
        return self._canonical_source.with_reactions_text(self._require_text(text, field_name="text"))

    def validate_source(self, source: MechanismAuthoringSource) -> ValidationResult:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")
        return self._validate_source(source)

    def validate_canonical(self) -> ValidationResult:
        if self._canonical_validation is None:
            self._canonical_validation = self._validate_source(self._canonical_source)
        return self._canonical_validation

    def source_is_ready_for_explicit_run(self, source: MechanismAuthoringSource) -> bool:
        validation = self.validate_source(source)
        if not validation.valid:
            return False
        if source.state_network_dsl.strip() and not self._topology_is_valid():
            return False
        return True

    def is_ready_for_explicit_run(self) -> bool:
        validation = self.validate_canonical()
        if not validation.valid:
            return False
        if self._canonical_source.state_network_dsl.strip() and not self._topology_is_valid():
            return False
        return True

    def is_ready_for_preview(self) -> bool:
        return self.is_ready_for_explicit_run()

    def explicit_run_source(self) -> MechanismAuthoringSource:
        if not self.is_ready_for_explicit_run():
            raise RuntimeError("Canonical mechanism is not ready for an explicit run.")
        return self._canonical_source

    def preview_source(self) -> MechanismAuthoringSource:
        if not self.is_ready_for_preview():
            raise RuntimeError("Mechanism source is not ready for preview.")
        return self._canonical_source

    def _validate_source(self, source: MechanismAuthoringSource) -> ValidationResult:
        validation_source = MechanismAuthoringSource.from_parts(
            reactions_text=_strip_named_initial_concentration_sets(source.reactions_text),
            state_network_dsl=source.state_network_dsl,
        )
        try:
            mechanism = parse_dsl_to_mechanism(validation_source.full_dsl, initials={})
            _ = apply_parameter_algebra_to_mechanism(
                validation_source.full_dsl,
                mechanism=mechanism,
                require_mutable=False,
            )
        except (DSLError, ValueError) as exc:
            return ValidationResult(False, str(exc), 0, 0, 0)
        return ValidationResult(
            valid=True,
            error_message="",
            species_count=len(mechanism.species),
            reaction_count=len(mechanism.reactions),
            equilibria_count=len(mechanism.equilibria),
        )

    def _topology_is_valid(self) -> bool:
        if self._topology_validator is None:
            return True
        return bool(self._topology_validator())

    @staticmethod
    def _require_text(value: str, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a str.")
        return value
