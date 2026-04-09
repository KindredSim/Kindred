from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kindred.core.batch_initial_conditions import (
    strip_named_reaction_dsl_initial_concentration_sets,
)
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError

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
        self._canonical_reactions_text = ""
        self._canonical_state_network_dsl = ""
        self._draft_reactions_text = ""
        self._draft_state_network_dsl = ""
        self._edit_session_active = False
        self._topology_validator = topology_validator
        self._canonical_validation: ValidationResult | None = None
        self._draft_validation: ValidationResult | None = None

    @property
    def canonical_reactions_text(self) -> str:
        return self._canonical_reactions_text

    @property
    def canonical_state_network_dsl(self) -> str:
        return self._canonical_state_network_dsl

    @property
    def canonical_full_dsl(self) -> str:
        return self._build_full_dsl(
            reactions_text=self._canonical_reactions_text,
            state_network_dsl=self._canonical_state_network_dsl,
        )

    @property
    def draft_reactions_text(self) -> str:
        if not self._edit_session_active:
            return ""
        return self._draft_reactions_text

    @property
    def draft_state_network_dsl(self) -> str:
        if not self._edit_session_active:
            return ""
        return self._draft_state_network_dsl

    @property
    def draft_full_dsl(self) -> str:
        return self._build_full_dsl(
            reactions_text=self.draft_reactions_text,
            state_network_dsl=self.draft_state_network_dsl,
        )

    @property
    def edit_session_active(self) -> bool:
        return self._edit_session_active

    def begin_edit_session(self) -> None:
        if self._edit_session_active:
            raise RuntimeError("Mechanism edit session is already active.")
        self._draft_reactions_text = self._canonical_reactions_text
        self._draft_state_network_dsl = self._canonical_state_network_dsl
        self._edit_session_active = True
        self._draft_validation = None

    def update_draft_reactions(self, text: str) -> None:
        self._require_active_edit_session()
        self._draft_reactions_text = self._require_text(text, field_name="text")
        self._draft_validation = None

    def update_draft_state_network(self, dsl: str) -> None:
        self._require_active_edit_session()
        self._draft_state_network_dsl = self._require_text(dsl, field_name="dsl")
        self._draft_validation = None

    def commit_edit_session(self) -> bool:
        self._require_active_edit_session()
        validation = self.validate_draft()
        if not validation.valid:
            return False
        if self._draft_state_network_dsl.strip() and not self._topology_is_valid():
            return False
        self._canonical_reactions_text = self._draft_reactions_text
        self._canonical_state_network_dsl = self._draft_state_network_dsl
        self._canonical_validation = validation
        self._end_edit_session()
        return True

    def cancel_edit_session(self) -> None:
        self._require_active_edit_session()
        self._end_edit_session()

    def apply_authoritative_update(self, reactions_text: str, state_network_dsl: str) -> None:
        canonical_reactions_text = self._require_text(
            reactions_text,
            field_name="reactions_text",
        )
        canonical_state_network_dsl = self._require_text(
            state_network_dsl,
            field_name="state_network_dsl",
        )
        self._canonical_reactions_text = canonical_reactions_text
        self._canonical_state_network_dsl = canonical_state_network_dsl
        self._canonical_validation = None
        if self._edit_session_active:
            self._end_edit_session()

    def validate_draft(self) -> ValidationResult:
        if self._draft_validation is None:
            self._draft_validation = self._validate_source(
                reactions_text=self.draft_reactions_text,
                state_network_dsl=self.draft_state_network_dsl,
            )
        return self._draft_validation

    def validate_canonical(self) -> ValidationResult:
        if self._canonical_validation is None:
            self._canonical_validation = self._validate_source(
                reactions_text=self._canonical_reactions_text,
                state_network_dsl=self._canonical_state_network_dsl,
            )
        return self._canonical_validation

    def is_ready_for_explicit_run(self) -> bool:
        validation = self.validate_canonical()
        if not validation.valid:
            return False
        if self._canonical_state_network_dsl.strip() and not self._topology_is_valid():
            return False
        return True

    def is_ready_for_preview(self) -> bool:
        if self._edit_session_active:
            return self.validate_draft().valid
        return self.is_ready_for_explicit_run()

    def explicit_run_source(self) -> str:
        if not self.is_ready_for_explicit_run():
            raise RuntimeError("Canonical mechanism is not ready for an explicit run.")
        return self.canonical_full_dsl

    def preview_source(self) -> str:
        if not self.is_ready_for_preview():
            raise RuntimeError("Mechanism source is not ready for preview.")
        if self._edit_session_active:
            return self.draft_full_dsl
        return self.canonical_full_dsl

    @staticmethod
    def _build_full_dsl(*, reactions_text: str, state_network_dsl: str) -> str:
        full_dsl = reactions_text
        state_network_dsl_s = state_network_dsl
        if state_network_dsl_s.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl_s
        return full_dsl

    def _validate_source(self, *, reactions_text: str, state_network_dsl: str) -> ValidationResult:
        normalized_dsl = self._build_simulation_source(
            reactions_text=reactions_text,
            state_network_dsl=state_network_dsl,
        )
        try:
            mechanism = parse_dsl_to_mechanism(normalized_dsl, initials={})
        except DSLError as exc:
            return ValidationResult(False, str(exc), 0, 0, 0)
        except Exception as exc:
            return ValidationResult(False, str(exc), 0, 0, 0)
        return ValidationResult(
            valid=True,
            error_message="",
            species_count=len(mechanism.species),
            reaction_count=len(mechanism.reactions),
            equilibria_count=len(mechanism.equilibria),
        )

    def _build_simulation_source(self, *, reactions_text: str, state_network_dsl: str) -> str:
        normalized_reactions = strip_named_reaction_dsl_initial_concentration_sets(
            reactions_text
        )
        return self._build_full_dsl(
            reactions_text=normalized_reactions,
            state_network_dsl=state_network_dsl,
        )

    def _require_active_edit_session(self) -> None:
        if not self._edit_session_active:
            raise RuntimeError("Mechanism edit session is not active.")

    def _topology_is_valid(self) -> bool:
        if self._topology_validator is None:
            return True
        return bool(self._topology_validator())

    @staticmethod
    def _require_text(value: str, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a str.")
        return value

    def _end_edit_session(self) -> None:
        self._draft_reactions_text = ""
        self._draft_state_network_dsl = ""
        self._draft_validation = None
        self._edit_session_active = False
