"""
Computational Simulator DSL tools.

Current contract
----------------
- Entries: `reaction:`, `equilibrium:` and a `States/TS` block.
- Accepted keys with normalization:
  T, degeneracy, κ, C°|C0, dG_act|ΔG‡, Ea, dG_eq|ΔG°, k|kf|kr, A, k_fast,
  energy=kcal/mol|kJ/mol|J/mol, state, edge
- If dG_eq/ΔG° is provided and Keq is omitted, compute Keq = exp(-ΔG°/(R*T)).
- Default per step is Eyring; Arrhenius override via A/Ea.
- Units inferred from molecularity (1/(M^(n−1)*s)); bimolecular Eyring divides by C°.
- TS degree fixed to 2 and enforced; attempts to violate raise a structured error.

Scope
-----
- Parse a simple line-oriented DSL (no nested syntax).
- Provide immutable analysis and retained prepared execution structures.
- No file access; caller passes text.

Non-goals
---------
- Persisting anything or solver integration.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC, Sequence as SequenceABC
import copy
import hashlib
import json
import logging
import math
import pickle
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Tuple, cast, TYPE_CHECKING

import numpy as np
import re

from ..equilibrium_rate_authority import (
    effective_reverse_rate_from_keq,
    validate_equilibrium_rate_authority_flags,
)
from ..units import UnitsModel
from .errors import (
    DSLError,
    invalid_number_error,
    missing_arrow_error,
    invalid_species_term_error,
    empty_stoichiometry_error,
    invalid_keyvalue_pair_error,
    invalid_boolean_error,
)
from .parameter_algebra_spec import classify_parameter_algebra_declaration

if TYPE_CHECKING:
    from ..intervention_schedule import InterventionSchedule
    from ..temperature import TemperatureScheduleProtocol
    from .state_model import StateNetwork

logger = logging.getLogger(__name__)


__all__ = [
    "DSLError",
    "DslAnalysis",
    "DslDerivedObservableDefinition",
    "DslDerivedScalarDefinition",
    "DslInitialConditionAdmission",
    "DslInitialConditionAssignment",
    "DslInitialConditionDeclaration",
    "DslParameterAnalysis",
    "DslParameterCoordinateUpdate",
    "DslSourceOrigin",
    "FrozenDslMapping",
    "PreparedDslRequestSet",
    "executed_parameter_analysis_from_mechanism",
    "derive_prepared_dsl_with_observables",
    "derive_prepared_dsl_with_parameter_updates",
    "prepare_dsl_request_set",
]


# ------------------------------ normalized keys ------------------------------

_KEY_ALIASES: Dict[str, str] = {
    # temperature / states
    "t": "T",
    "degeneracy": "degeneracy",
    "kappa": "κ",
    "κ": "κ",
    "c°": "C0",
    "c0": "C0",
    # energies / model params
    "dg_act": "dG_act",
    "Δg‡": "dG_act",
    "Δg^‡": "dG_act",
    "dG_act": "dG_act",
    "ΔG‡": "dG_act",
    "ea": "Ea",
    "Ea": "Ea",
    "dg_eq": "dG_eq",
    "Δg°": "dG_eq",
    "dG_eq": "dG_eq",
    "ΔG°": "dG_eq",
    "a": "A",
    "A": "A",
    "k": "k",
    "keq": "Keq",
    "kf": "kf",
    "kr": "kr",
    "k_fast": "k_fast",
    # units selector
    "energy": "energy",
    # state graph
    "state": "state",
    "edge": "edge",
}

_ARROW_RE = re.compile(r"<->|<=>|->|=>")
_SPECIES_TERM_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*\*?\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*$")
_COMMA_SEMI_SPLIT_RE = re.compile(r"[,;]")
_STATE_REST_SPLIT_RE = re.compile(r"[;,]")
_SEMI_SPLIT_RE = re.compile(r"[;]")

_REACTION_KNOWN_KEYS = frozenset({"κ", "kf", "k", "kr", "A", "Ea", "Keq", "dG_eq", "dG_act"})
_EQUILIBRIUM_KNOWN_KEYS = frozenset({"Keq", "kf", "kr", "dG_eq", "dg_eq", "cm_id", "cm_std_ratio"})


# ------------------------------ data models ----------------------------------

@dataclass
class DSLIR:
    """
    Structured intermediate representation (IR) for the simulator DSL.

    Immutable analysis and Mechanism construction are derived from this IR to
    prevent semantic drift across entry points.
    """

    version: int
    energy_unit: str
    temperature_K: float
    standard_conc_M: float
    kappa_global: float
    temperature_schedule: Optional["TemperatureScheduleProtocol"]
    intervention_schedule: Optional["InterventionSchedule"]
    state_network: "StateNetwork"
    state_network_facts: Dict[str, object]
    steps: List["ParsedStep"]
    algebra_lines: List[str]
    inline_initial_conditions: "DslInitialConditionAdmission"
    notes: List[str] = field(default_factory=list)


class FrozenDslMapping(MappingABC[str, object]):
    """Pickle-safe deeply immutable mapping used by public DSL evidence."""

    __slots__ = ("__items",)

    def __init__(
        self,
        values: Mapping[str, object] | tuple[tuple[str, object], ...] = (),
    ) -> None:
        items = (
            tuple(values.items())
            if isinstance(values, MappingABC)
            else tuple(values)
        )
        normalized: list[tuple[str, object]] = []
        seen: set[str] = set()
        for key, value in items:
            if type(key) is not str:
                raise TypeError("Frozen DSL mapping keys must be exact strings.")
            if key in seen:
                raise ValueError(f"Duplicate frozen DSL mapping key: {key!r}.")
            seen.add(key)
            normalized.append((key, value))
        self.__items = tuple(normalized)

    def __getitem__(self, key: str) -> object:
        for candidate, value in self.__items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self.__items)

    def __len__(self) -> int:
        return len(self.__items)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (type(self), (self.__items,))


def _canonical_dsl_value(
    value: object,
    *,
    active_ids: set[int] | None = None,
) -> object:
    from kindred.core.intervention_schedule import InterventionSchedule
    from kindred.core.temperature import TemperatureSchedule

    if value is None or type(value) in {str, bool, int, bytes}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("DSL analysis values must be finite.")
        return value
    if type(value) is TemperatureSchedule:
        return _canonical_dsl_value(value.to_dict(), active_ids=active_ids)
    if type(value) is InterventionSchedule:
        return _canonical_dsl_value(value.to_payload(), active_ids=active_ids)
    if isinstance(value, np.generic):
        return _canonical_dsl_value(value.item(), active_ids=active_ids)
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError("DSL analysis arrays cannot contain objects.")
        return _canonical_dsl_value(value.tolist(), active_ids=active_ids)
    if active_ids is None:
        active_ids = set()
    if isinstance(value, MappingABC):
        identity = id(value)
        if identity in active_ids:
            raise TypeError("DSL analysis values cannot contain cycles.")
        active_ids.add(identity)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        "DSL analysis mapping keys must be exact strings."
                    )
                result[key] = _canonical_dsl_value(
                    item,
                    active_ids=active_ids,
                )
            return result
        finally:
            active_ids.remove(identity)
    if isinstance(value, SequenceABC) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        identity = id(value)
        if identity in active_ids:
            raise TypeError("DSL analysis values cannot contain cycles.")
        active_ids.add(identity)
        try:
            return [
                _canonical_dsl_value(item, active_ids=active_ids)
                for item in value
            ]
        finally:
            active_ids.remove(identity)
    raise TypeError(
        "DSL analysis accepts only explicit code-owned schedules and exact "
        "JSON-safe values."
    )


def _freeze_dsl_value(
    value: object,
    *,
    active_ids: set[int] | None = None,
) -> object:
    if value is None or type(value) in {str, bool, int, float, bytes}:
        return value
    if isinstance(value, (DslAnalysis, DslParameterAnalysis, DslSourceOrigin)):
        return value
    if active_ids is None:
        active_ids = set()
    if isinstance(value, MappingABC):
        identity = id(value)
        if identity in active_ids:
            raise TypeError("DSL immutable values cannot contain cycles.")
        active_ids.add(identity)
        try:
            return FrozenDslMapping(
                tuple(
                    (
                        str(key),
                        _freeze_dsl_value(item, active_ids=active_ids),
                    )
                    for key, item in value.items()
                )
            )
        finally:
            active_ids.remove(identity)
    if isinstance(value, SequenceABC) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        identity = id(value)
        if identity in active_ids:
            raise TypeError("DSL immutable values cannot contain cycles.")
        active_ids.add(identity)
        try:
            return tuple(
                _freeze_dsl_value(item, active_ids=active_ids)
                for item in value
            )
        finally:
            active_ids.remove(identity)
    raise TypeError(f"Unsupported DSL immutable value: {type(value).__name__}.")


def _dsl_identity_payload(
    text: str,
    units: UnitsModel | None,
) -> dict[str, object]:
    if type(text) is not str:
        raise TypeError("DSL source must be an exact string.")
    if units is not None and type(units) is not UnitsModel:
        raise TypeError("DSL units must be UnitsModel or None.")
    effective = UnitsModel() if units is None else units
    return {
        "dsl": "\n".join(line.rstrip() for line in text.splitlines()).strip(),
        "energy_unit": effective.energy_unit,
        "temperature_K": effective.temperature_K,
        "standard_conc_M": effective.standard_conc_M,
    }


@dataclass(frozen=True, slots=True)
class DslSourceOrigin:
    authored_ordinal: int
    line_number: int
    kind: str
    source_text: str

    def __post_init__(self) -> None:
        if (
            type(self.authored_ordinal) is not int
            or self.authored_ordinal < 0
        ):
            raise ValueError("DSL origin ordinal must be a non-negative int.")
        if type(self.line_number) is not int or self.line_number <= 0:
            raise ValueError("DSL origin line number must be a positive int.")
        if type(self.kind) is not str or not self.kind:
            raise ValueError("DSL origin kind must be nonempty text.")
        if type(self.source_text) is not str:
            raise TypeError("DSL origin source text must be an exact string.")


@dataclass(frozen=True, slots=True)
class DslParameterAnalysis:
    name: str
    value: float
    context: str
    source: str
    step_index: int | None = None
    editable: bool = True
    unit: str | None = None
    role: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("DSL parameter name must be nonempty text.")
        if type(self.value) not in {int, float}:
            raise TypeError("DSL parameter values must be exact numbers.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("DSL parameter values must be finite.")
        object.__setattr__(self, "value", value)
        for attr in ("context", "source"):
            if type(getattr(self, attr)) is not str:
                raise TypeError(
                    f"DSL parameter {attr} must be an exact string."
                )
        if self.step_index is not None and type(self.step_index) is not int:
            raise TypeError(
                "DSL parameter step index must be an exact int or None."
            )
        if type(self.editable) is not bool:
            raise TypeError("DSL parameter editable must be bool.")
        for attr in ("unit", "role"):
            raw = getattr(self, attr)
            if raw is not None and type(raw) is not str:
                raise TypeError(
                    f"DSL parameter {attr} must be a string or None."
                )
        for attr in ("minimum", "maximum"):
            raw = getattr(self, attr)
            if raw is None:
                continue
            if type(raw) not in {int, float}:
                raise TypeError(
                    f"DSL parameter {attr} must be an exact number."
                )
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise ValueError(f"DSL parameter {attr} must be finite.")
            object.__setattr__(self, attr, numeric)


@dataclass(frozen=True, slots=True)
class DslParameterCoordinateUpdate:
    """One exact writable coordinate value for typed DSL derivation."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError(
                "DSL parameter update name must be an exact nonempty string."
            )
        if type(self.value) not in {int, float}:
            raise TypeError("DSL parameter update value must be an exact number.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("DSL parameter update value must be finite.")
        object.__setattr__(self, "value", value)


_DSL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class DslInitialConditionAssignment:
    """One validated, origin-stamped inline concentration assignment."""

    species: str
    value: float
    authored_ordinal: int
    line_number: int
    source_text: str

    def __post_init__(self) -> None:
        if (
            type(self.species) is not str
            or not _DSL_IDENTIFIER_RE.fullmatch(self.species)
        ):
            raise ValueError(
                "Inline initial-condition species must be an identifier."
            )
        if type(self.value) not in {int, float}:
            raise TypeError(
                "Inline initial-condition values must be exact numbers."
            )
        value = float(self.value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "Inline initial-condition values must be finite and "
                "nonnegative."
            )
        if (
            type(self.authored_ordinal) is not int
            or self.authored_ordinal < 0
        ):
            raise ValueError(
                "Inline initial-condition assignment ordinal must be "
                "non-negative."
            )
        if type(self.line_number) is not int or self.line_number <= 0:
            raise ValueError(
                "Inline initial-condition assignment line must be positive."
            )
        if type(self.source_text) is not str:
            raise TypeError(
                "Inline initial-condition assignment source must be text."
            )
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class DslInitialConditionDeclaration:
    """One named block or one anonymous source-ordered declaration."""

    source_kind: str
    source_name: str | None
    assignments: tuple[DslInitialConditionAssignment, ...]
    start_line: int
    end_line: int
    source_lines: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_kind not in {"named", "anonymous"}:
            raise ValueError(
                "Inline initial-condition kind must be named or anonymous."
            )
        if self.source_kind == "named":
            if (
                type(self.source_name) is not str
                or not _DSL_IDENTIFIER_RE.fullmatch(self.source_name)
            ):
                raise ValueError(
                    "Named initial-condition set names must be identifiers."
                )
        elif self.source_name is not None:
            raise ValueError(
                "Anonymous initial-condition declarations have no source name."
            )
        if type(self.assignments) is not tuple or any(
            type(item) is not DslInitialConditionAssignment
            for item in self.assignments
        ):
            raise TypeError(
                "Inline initial-condition assignments must be an exact tuple."
            )
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line <= 0
            or self.end_line < self.start_line
        ):
            raise ValueError(
                "Inline initial-condition source span must be ordered."
            )
        if (
            type(self.source_lines) is not tuple
            or len(self.source_lines) != self.end_line - self.start_line + 1
            or any(type(line) is not str for line in self.source_lines)
        ):
            raise TypeError(
                "Inline initial-condition source lines must match the span."
            )

    @property
    def values(self) -> Mapping[str, float]:
        values: dict[str, float] = {}
        for assignment in self.assignments:
            values[assignment.species] = assignment.value
        return cast(Mapping[str, float], FrozenDslMapping(values))


@dataclass(frozen=True, slots=True)
class DslInitialConditionAdmission:
    """The sole typed inline-IC parser result for one exact source."""

    declarations: tuple[DslInitialConditionDeclaration, ...] = ()
    materialized_source: str = ""

    def __post_init__(self) -> None:
        if type(self.declarations) is not tuple or any(
            type(item) is not DslInitialConditionDeclaration
            for item in self.declarations
        ):
            raise TypeError(
                "Inline initial-condition declarations must be an exact tuple."
            )
        if type(self.materialized_source) is not str:
            raise TypeError(
                "Materialized inline initial-condition source must be text."
            )
        names = [
            declaration.source_name.casefold()
            for declaration in self.declarations
            if declaration.source_name is not None
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                "Inline initial-condition set names must be unique."
            )

    @property
    def has_declarations(self) -> bool:
        return bool(self.declarations)

    @property
    def named_declarations(
        self,
    ) -> tuple[DslInitialConditionDeclaration, ...]:
        return tuple(
            item for item in self.declarations if item.source_kind == "named"
        )

    @property
    def anonymous_assignments(
        self,
    ) -> tuple[DslInitialConditionAssignment, ...]:
        return tuple(
            assignment
            for item in self.declarations
            if item.source_kind == "anonymous"
            for assignment in item.assignments
        )


@dataclass(frozen=True, slots=True)
class DslDerivedScalarDefinition:
    name: str
    value: float
    minimum: float = float("-inf")
    maximum: float = float("inf")
    scope: str = "shared"

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        value = float(self.value)
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        scope = str(self.scope or "").strip().lower()
        if not _DSL_IDENTIFIER_RE.fullmatch(name):
            raise ValueError(
                "Derived DSL scalar name must be an identifier."
            )
        if (
            not math.isfinite(value)
            or math.isnan(minimum)
            or math.isnan(maximum)
            or minimum >= maximum
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"Derived DSL scalar {name!r} requires a finite value "
                "inside ordered non-NaN bounds."
            )
        if scope not in {"shared", "per_dataset"}:
            raise ValueError(
                "Derived DSL scalar scope must be shared or per_dataset."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class DslDerivedObservableDefinition:
    name: str
    expression: str
    scalar_parameters: tuple[DslDerivedScalarDefinition, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        expression = str(self.expression or "").strip()
        scalars = tuple(self.scalar_parameters)
        if not _DSL_IDENTIFIER_RE.fullmatch(name):
            raise ValueError(
                "Derived DSL observable name must be an identifier."
            )
        if not expression:
            raise ValueError(
                "Derived DSL observable expression must be nonblank."
            )
        if any(
            type(item) is not DslDerivedScalarDefinition
            for item in scalars
        ):
            raise TypeError(
                "Derived DSL observable scalars must be exact "
                "DslDerivedScalarDefinition values."
            )
        scalar_names = tuple(item.name for item in scalars)
        if len(scalar_names) != len(set(scalar_names)):
            raise ValueError(
                "Derived DSL observable scalar names must be unique."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "scalar_parameters", scalars)


@dataclass(frozen=True, slots=True)
class DslAnalysis:
    """Deeply immutable, origin-stamped public DSL admission evidence."""

    identity: str
    normalized_text: str
    origins: tuple[DslSourceOrigin, ...]
    species_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    parameters: tuple[DslParameterAnalysis, ...]
    inline_initial_conditions: DslInitialConditionAdmission
    temperature_schedule: object
    intervention_schedule: object
    structure: Mapping[str, object] = field(
        default_factory=FrozenDslMapping
    )

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not str
            or not self.identity
            or self.identity != self.identity.strip()
        ):
            raise ValueError("DSL analysis identity must be nonempty text.")
        if type(self.normalized_text) is not str:
            raise TypeError("Normalized DSL text must be an exact string.")
        if type(self.origins) is not tuple or any(
            type(item) is not DslSourceOrigin for item in self.origins
        ):
            raise TypeError(
                "DSL origins must be an exact tuple of DslSourceOrigin values."
            )
        if tuple(item.authored_ordinal for item in self.origins) != tuple(
            range(len(self.origins))
        ):
            raise ValueError("DSL origin ordinals must be contiguous.")
        for attr, label in (
            ("species_names", "species"),
            ("observable_names", "observable"),
        ):
            values = getattr(self, attr)
            if type(values) is not tuple or any(
                type(item) is not str
                or not item
                or item != item.strip()
                for item in values
            ):
                raise TypeError(
                    f"DSL {label} names must be an exact tuple of "
                    "nonempty strings."
                )
            if len(set(values)) != len(values):
                raise ValueError(f"DSL {label} names must be unique.")
        if set(self.species_names) & set(self.observable_names):
            raise ValueError(
                "DSL observable names must not duplicate model species names."
            )
        if type(self.parameters) is not tuple or any(
            type(item) is not DslParameterAnalysis
            for item in self.parameters
        ):
            raise TypeError(
                "DSL analysis parameters must be DslParameterAnalysis values."
            )
        if len({item.name for item in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("DSL analysis parameter names must be unique.")
        if type(self.inline_initial_conditions) is not DslInitialConditionAdmission:
            raise TypeError(
                "DSL inline initial conditions must be one typed admission."
            )
        for attr in ("temperature_schedule", "intervention_schedule"):
            canonical = _canonical_dsl_value(getattr(self, attr))
            object.__setattr__(self, attr, _freeze_dsl_value(canonical))
        canonical_structure = _canonical_dsl_value(self.structure)
        frozen_structure = _freeze_dsl_value(canonical_structure)
        if not isinstance(frozen_structure, FrozenDslMapping):
            raise TypeError("DSL structure must be a mapping.")
        object.__setattr__(self, "structure", frozen_structure)

    @property
    def parameter_values(self) -> Mapping[str, float]:
        return cast(
            Mapping[str, float],
            FrozenDslMapping(
                {item.name: item.value for item in self.parameters}
            ),
        )


class PreparedDslRequestSet:
    """Single-parse owner of one immutable analysis and execution structure."""

    __slots__ = (
        "analysis",
        "__mechanism",
        "__intervention_schedule",
        "__parameter_algebra_spec",
    )

    def __init__(
        self,
        *,
        analysis: DslAnalysis,
        mechanism: object,
        intervention_schedule: object,
        parameter_algebra_spec: object,
    ) -> None:
        if type(analysis) is not DslAnalysis:
            raise TypeError("analysis must be DslAnalysis.")
        self.analysis = analysis
        self.__mechanism = mechanism
        self.__intervention_schedule = intervention_schedule
        self.__parameter_algebra_spec = parameter_algebra_spec

    def mechanism_for_execution(self) -> object:
        return copy.deepcopy(self.__mechanism)

    def intervention_schedule_for_execution(self) -> object:
        return copy.deepcopy(self.__intervention_schedule)

    def parameter_algebra_spec_for_execution(self) -> object:
        return copy.deepcopy(self.__parameter_algebra_spec)

    def process_execution_payload(
        self,
        *,
        wegscheider_cyclicity_enabled: bool,
    ) -> Mapping[str, object]:
        from kindred.core.mechanism_metadata import MechanismMetadataKeys
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )
        from kindred.core.simulator.wegscheider_symbolic import (
            analyze_wegscheider_cyclicity,
        )

        mechanism = self.mechanism_for_execution()
        metadata = getattr(mechanism, "metadata", None)
        if not isinstance(metadata, dict):
            raise TypeError("Prepared DSL mechanism metadata must be mutable.")
        metadata[
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED
        ] = bool(wegscheider_cyclicity_enabled)
        apply_parameter_algebra_spec_to_mechanism(
            self.parameter_algebra_spec_for_execution(),
            mechanism=mechanism,
            require_mutable=True,
        )
        if bool(wegscheider_cyclicity_enabled) and bool(
            getattr(mechanism, "equilibria", ()) or ()
        ):
            report = analyze_wegscheider_cyclicity(mechanism)
            if report.cycles:
                metadata["symbolic_wegscheider_identity"] = dict(
                    report.symbolic_identity
                )

        species_names = tuple(str(name) for name in mechanism.species_names())
        schedule = self.intervention_schedule_for_execution()
        schedule_payload = (
            schedule.to_payload()
            if callable(getattr(schedule, "to_payload", None))
            else schedule
        )
        payload = _freeze_dsl_value(
            {
                "version": 4,
                "dsl_analysis": self.analysis,
                "mechanism_blob": pickle.dumps(
                    mechanism,
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
                "species_names": species_names,
                "y0": tuple(
                    float(mechanism.species[name].initial_conc)
                    for name in species_names
                ),
                "temperature_schedule": self.analysis.temperature_schedule,
                "intervention_schedule": schedule_payload,
                "unresolved_intervention_schedule": schedule_payload,
                "jacobian_func": None,
            }
        )
        if not isinstance(payload, FrozenDslMapping):
            raise TypeError(
                "Prepared DSL execution payload must be a mapping."
            )
        return payload


# ------------------------------ helpers --------------------------------------

def _norm_key(k: str) -> str:
    key = k.strip()
    if key.lower() == "keq":
        return "Keq"
    # Bare uppercase K is not lowercase k and is not accepted as a Keq alias.
    if key == "K":
        return "K"
    # Otherwise normalize to lowercase and look up
    return _KEY_ALIASES.get(key.lower(), key)


def _parse_keyvals(
    rest: str,
    *,
    reject_duplicate_canonical_keys: bool = False,
    line_number: int | None = None,
    line_content: str | None = None,
) -> Dict[str, str]:
    """
    Parse comma or semicolon-separated key=value pairs with lenient spacing.

    Example: "A=1.2e10, Ea=50, energy=kJ/mol" -> {"A":"1.2e10","Ea":"50","energy":"kJ/mol"}
    Example: "kf=1.5; kr=0.25" -> {"kf":"1.5","kr":"0.25"}
    """
    out: Dict[str, str] = {}
    original_spellings: Dict[str, str] = {}
    if not rest.strip():
        return out
    # Split on both commas and semicolons to support both formats
    for chunk in _COMMA_SEMI_SPLIT_RE.split(rest):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            # tolerate stray tokens like "state=..." lines; raise for truly malformed
            raise invalid_keyvalue_pair_error(chunk)
        k, v = chunk.split("=", 1)
        raw_key = k.strip()
        canonical_key = _norm_key(raw_key)
        previous_spelling = original_spellings.get(canonical_key)
        if reject_duplicate_canonical_keys and previous_spelling is not None:
            raise DSLError(
                f"Duplicate parameter: '{previous_spelling}' and '{raw_key}' both resolve to {canonical_key}",
                line_number=line_number,
                line_content=line_content,
            )
        original_spellings[canonical_key] = raw_key
        out[canonical_key] = v.strip()
    return out


_MEMBERS_TERM_RE = re.compile(r"^(\d+)?([A-Za-z_][A-Za-z0-9_]*)$")


def _parse_members_expr(expr: str) -> Tuple[str, ...]:
    """
    Parse a state `members=` expression like "A+B" or "2A+B" into a tuple with duplicates.

    This is used to recover stoichiometry for state-network conversion.
    """
    s = str(expr or "").strip()
    if not s:
        raise ValueError("members cannot be empty")
    parts = [p.strip() for p in s.split("+") if p.strip()]
    if not parts:
        raise ValueError("members cannot be empty")
    out: List[str] = []
    for raw in parts:
        term = raw.replace(" ", "")
        m = _MEMBERS_TERM_RE.match(term)
        if not m:
            raise ValueError(f"invalid members term {raw!r}")
        coeff = int(m.group(1) or "1")
        if coeff <= 0:
            raise ValueError("members coefficients must be positive integers")
        name = m.group(2)
        out.extend([name] * coeff)
    return tuple(out)


def _parse_state_rest(rest: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Parse the `state:` line body and return (name, kv).

    Supports both:
      - "name=A; kind=GS; energy=0.0"
      - "A, kind=GS, energy=0.0"
    """
    parts = [p.strip() for p in _STATE_REST_SPLIT_RE.split(str(rest or "")) if p.strip()]
    name: Optional[str] = None
    kv_parts: List[str] = []
    for part in parts:
        if "=" in part:
            kv_parts.append(part)
        elif name is None:
            name = part
    kv = _parse_keyvals(",".join(kv_parts)) if kv_parts else {}
    if not name:
        name = kv.get("state") or kv.get("name")
    return name, kv


def _extract_numeric_value(expr: str) -> Optional[float]:
    """
    Best-effort extraction of a numeric literal from a DSL value string.

    Accepts values like "1.0", "1e-3", or "12.5 kJ/mol" (unit is ignored).
    Returns None if the value is not purely numeric.
    """
    if expr is None:
        return None
    value = expr.strip()
    if not value:
        return None
    parts = value.split()
    candidate = parts[0]
    return _float_or_none(candidate)


def _parse_species_side(text: str) -> Dict[str, float]:
    """
    Parse a side like "A + 2B + 0.5*C" into {name: coeff}.
    Coefficients default to 1.0 when omitted.
    """
    if not text.strip():
        return {}
    result: Dict[str, float] = {}
    for term in text.split("+"):
        term = term.strip()
        if not term:
            raise DSLError("Malformed stoichiometry: empty term (e.g., consecutive '+' signs)")
        m = _SPECIES_TERM_RE.match(term)
        if not m and "*" in term:
            # Allow patterns like "2 * B" while rejecting "A B"
            m = _SPECIES_TERM_RE.match(term.replace(" ", ""))
        if not m:
            raise invalid_species_term_error(term)
        coef_s, name = m.groups()
        coef = _float_or_none(coef_s) if coef_s is not None else 1.0
        if coef == 0.0:
            continue
        result[name] = result.get(name, 0.0) + coef
    if not result:
        raise empty_stoichiometry_error()
    return result


def _parse_stoich(ar: str) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Parse "A + B -> C" or "2A <-> B + C" into reactants, products, arrow.
    Returns (reactants, products, arrow).
    """
    m = _ARROW_RE.search(ar)
    if not m:
        raise missing_arrow_error(ar)
    arrow = m.group(0)
    lhs = ar[: m.start()]
    rhs = ar[m.end() :]
    react = _parse_species_side(lhs)  # allow "2*B"
    prod = _parse_species_side(rhs)
    return react, prod, arrow


def _float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(s)
    except (TypeError, ValueError, OverflowError):
        logger.debug(f"Failed to convert '{s}' to float", exc_info=True)
        raise invalid_number_error(str(s), "value")
    if not math.isfinite(v):
        raise invalid_number_error(str(s), "value")
    return v


def _validate_rate_or_K(
    value: Optional[float],
    name: str,
    *,
    line_number: int | None = None,
    line_content: str | None = None,
) -> None:
    if value is None:
        return
    if math.isnan(value) or math.isinf(value) or value < 0:
        raise DSLError(
            f"'{name}' must be a non-negative finite number, got {value}",
            line_number=line_number,
            line_content=line_content,
        )


def _bool_from_str(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise invalid_boolean_error(s, "boolean")


def _derive_fast_equilibrium_rates_with_context(
    *,
    line_number: Optional[int],
    line_content: Optional[str],
    **kwargs,
):
    """
    Translate fast-equilibrium authority errors into source-aware DSL errors.
    """
    try:
        from ..equilibrium_rate_authority import (
            derive_fast_equilibrium_rates,
        )

        if "T" in kwargs:
            kwargs["temperature_K"] = kwargs.pop("T")
        return derive_fast_equilibrium_rates(**kwargs)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc


def _validate_equilibrium_rate_authority_for_dsl(
    *,
    has_kf: bool,
    has_kr: bool,
    has_Keq: bool,
    has_dG_eq: bool,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    try:
        validate_equilibrium_rate_authority_flags(
            has_kf=has_kf,
            has_kr=has_kr,
            has_Keq=has_Keq,
            has_dG_eq=has_dG_eq,
        )
    except ValueError as exc:
        raise DSLError(
            str(exc),
            suggestion="Provide kf=... with either kr=... or Keq=/dG_eq=..., but not both.",
            examples=[
                "equilibrium: A <-> B; kf=4.0; kr=2.0",
                "equilibrium: A <-> B; kf=10.0; Keq=5.0",
                "equilibrium: A <-> B; kf=10.0; dG_eq=-4.0",
            ],
            line_number=line_number,
            line_content=line_content,
        ) from exc


def _forward_rate_family_from_params(
    params: Mapping[str, str],
    *,
    line_number: int | None,
    line_content: str | None,
) -> str:
    """Select exactly one authored Constant/Arrhenius/Eyring family."""

    if "kf" in params and "k" in params:
        raise DSLError(
            "Forward rate authority cannot contain both k and kf.",
            line_number=line_number,
            line_content=line_content,
        )
    families = {
        family
        for family, present in (
            ("Constant", any(key in params for key in ("kf", "k"))),
            ("Arrhenius", any(key in params for key in ("A", "Ea"))),
            ("Eyring", "dG_act" in params),
        )
        if present
    }
    if len(families) > 1:
        raise DSLError(
            "Forward rate authority must use exactly one of Constant "
            "(k or kf), Arrhenius (A and Ea), or Eyring (dG_act).",
            line_number=line_number,
            line_content=line_content,
        )
    family = next(iter(families), "Eyring")
    if "κ" in params and family != "Eyring":
        raise DSLError(
            "Per-step κ is valid only for an Eyring forward rate.",
            line_number=line_number,
            line_content=line_content,
        )
    return family


# ------------------------------ main parsing ---------------------------------

_DSL_IR_VERSION = 1
_NAMED_INITIAL_BLOCK_START_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{\s*(?P<body>.*?)\s*$"
)
_BRACKET_INITIAL_ASSIGNMENT_RE = re.compile(
    r"^\[(?P<species>[A-Za-z_][A-Za-z0-9_]*)\]\s*=\s*(?P<value>.+?)\s*$"
)
_INITIAL_PROVENANCE_PREFIX = "# Kindred IC provenance | "


def _inline_initial_number(
    token: str,
    *,
    set_name: str | None,
    species: str,
    line_number: int,
    line_content: str,
) -> float:
    context = (
        f"set {set_name!r}, species {species!r}"
        if set_name is not None
        else f"anonymous set, species {species!r}"
    )
    try:
        value = float(str(token).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise DSLError(
            f"Initial condition value for {context} must be numeric.",
            line_number=line_number,
            line_content=line_content,
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise DSLError(
            f"Initial condition value for {context} must be finite and "
            "nonnegative.",
            line_number=line_number,
            line_content=line_content,
        )
    return value


def _inline_initial_assignments(
    code: str,
    *,
    set_name: str | None,
    line_number: int,
    line_content: str,
    first_ordinal: int,
) -> tuple[DslInitialConditionAssignment, ...]:
    stripped = str(code).strip()
    lowered = stripped.lower()
    if lowered.startswith(("init:", "initial:")):
        _prefix, body = stripped.split(":", 1)
        raw_parts = body.split(",")
        if not body.strip() or any(not part.strip() for part in raw_parts):
            raise DSLError(
                "Initial condition directive requires comma-separated "
                "species=value assignments.",
                line_number=line_number,
                line_content=line_content,
            )
        pairs: list[tuple[str, str]] = []
        for part in raw_parts:
            if part.count("=") != 1:
                raise DSLError(
                    "Initial condition directive assignments must use exactly "
                    "one '='.",
                    line_number=line_number,
                    line_content=line_content,
                )
            species, value_token = part.split("=", 1)
            species = species.strip()
            if not _DSL_IDENTIFIER_RE.fullmatch(species):
                raise DSLError(
                    f"Initial condition species {species!r} must be an "
                    "identifier.",
                    line_number=line_number,
                    line_content=line_content,
                )
            pairs.append((species, value_token))
    else:
        raw_parts = stripped.split(",")
        if any(not part.strip() for part in raw_parts):
            raise DSLError(
                "Bracket initial conditions must be comma-separated "
                "assignments.",
                line_number=line_number,
                line_content=line_content,
            )
        pairs = []
        for part in raw_parts:
            match = _BRACKET_INITIAL_ASSIGNMENT_RE.fullmatch(part.strip())
            if match is None:
                raise DSLError(
                    "Initial conditions must use '[Species] = value' or "
                    "'init: Species=value' syntax.",
                    line_number=line_number,
                    line_content=line_content,
                )
            pairs.append(
                (
                    str(match.group("species")),
                    str(match.group("value")),
                )
            )
    return tuple(
        DslInitialConditionAssignment(
            species=species,
            value=_inline_initial_number(
                value_token,
                set_name=set_name,
                species=species,
                line_number=line_number,
                line_content=line_content,
            ),
            authored_ordinal=first_ordinal + index,
            line_number=line_number,
            source_text=line_content,
        )
        for index, (species, value_token) in enumerate(pairs)
    )


def _commented_initial_source(
    lines: list[str],
    claimed_indexes: set[int],
) -> str:
    rendered = [
        (
            _INITIAL_PROVENANCE_PREFIX + line.rstrip()
            if index in claimed_indexes
            else line.rstrip()
        )
        for index, line in enumerate(lines)
    ]
    return "\n".join(rendered).strip()


def _admit_inline_initial_conditions(
    text: str,
) -> tuple[DslInitialConditionAdmission, frozenset[int]]:
    """Parse all inline IC syntax once and return claimed zero-based lines."""

    if type(text) is not str:
        raise TypeError("DSL source must be an exact string.")
    lines = text.splitlines()
    declarations: list[DslInitialConditionDeclaration] = []
    claimed_indexes: set[int] = set()
    named_keys: set[str] = set()
    assignment_ordinal = 0
    index = 0
    while index < len(lines):
        raw = lines[index]
        code = raw.split("#", 1)[0].strip()
        if not code:
            index += 1
            continue
        header = _NAMED_INITIAL_BLOCK_START_RE.fullmatch(code)
        if header is not None:
            set_name = str(header.group("name"))
            body = str(header.group("body"))
            # Parameter algebra declarations own their own grammar.
            if set_name.lower() in {"let", "param"}:
                index += 1
                continue
            key = set_name.casefold()
            if key in named_keys:
                raise DSLError(
                    f"Duplicate inline initial-condition set name: "
                    f"{set_name!r}.",
                    line_number=index + 1,
                    line_content=raw,
                )
            named_keys.add(key)
            start = index
            assignments: list[DslInitialConditionAssignment] = []
            one_line_closed = False
            if "}" in body:
                before_close, close, after_close = body.partition("}")
                if close and after_close.strip():
                    raise DSLError(
                        "Inline initial-condition block closing '}' must end "
                        "the declaration.",
                        line_number=index + 1,
                        line_content=raw,
                    )
                if before_close.strip():
                    parsed = _inline_initial_assignments(
                        before_close.strip(),
                        set_name=set_name,
                        line_number=index + 1,
                        line_content=raw,
                        first_ordinal=assignment_ordinal,
                    )
                    assignments.extend(parsed)
                    assignment_ordinal += len(parsed)
                one_line_closed = True
            elif body.strip():
                parsed = _inline_initial_assignments(
                    body,
                    set_name=set_name,
                    line_number=index + 1,
                    line_content=raw,
                    first_ordinal=assignment_ordinal,
                )
                assignments.extend(parsed)
                assignment_ordinal += len(parsed)
            if one_line_closed:
                end = index
            else:
                cursor = index + 1
                while cursor < len(lines):
                    block_raw = lines[cursor]
                    block_code = block_raw.split("#", 1)[0].strip()
                    if not block_code:
                        cursor += 1
                        continue
                    if block_code == "}":
                        break
                    if "}" in block_code:
                        before_close, close, after_close = block_code.partition(
                            "}"
                        )
                        if not close or after_close.strip():
                            raise DSLError(
                                "Inline initial-condition block closing '}' "
                                "must end the declaration.",
                                line_number=cursor + 1,
                                line_content=block_raw,
                            )
                        if before_close.strip():
                            parsed = _inline_initial_assignments(
                                before_close,
                                set_name=set_name,
                                line_number=cursor + 1,
                                line_content=block_raw,
                                first_ordinal=assignment_ordinal,
                            )
                            assignments.extend(parsed)
                            assignment_ordinal += len(parsed)
                        break
                    parsed = _inline_initial_assignments(
                        block_code,
                        set_name=set_name,
                        line_number=cursor + 1,
                        line_content=block_raw,
                        first_ordinal=assignment_ordinal,
                    )
                    assignments.extend(parsed)
                    assignment_ordinal += len(parsed)
                    cursor += 1
                if cursor >= len(lines):
                    raise DSLError(
                        f"Inline initial-condition set {set_name!r} is "
                        "missing a closing '}'.",
                        line_number=start + 1,
                        line_content=raw,
                    )
                end = cursor
            claimed_indexes.update(range(start, end + 1))
            declarations.append(
                DslInitialConditionDeclaration(
                    source_kind="named",
                    source_name=set_name,
                    assignments=tuple(assignments),
                    start_line=start + 1,
                    end_line=end + 1,
                    source_lines=tuple(lines[start : end + 1]),
                )
            )
            index = end + 1
            continue
        lowered = code.lower()
        if code.startswith("[") or lowered.startswith(("init:", "initial:")):
            parsed = _inline_initial_assignments(
                code,
                set_name=None,
                line_number=index + 1,
                line_content=raw,
                first_ordinal=assignment_ordinal,
            )
            assignment_ordinal += len(parsed)
            claimed_indexes.add(index)
            declarations.append(
                DslInitialConditionDeclaration(
                    source_kind="anonymous",
                    source_name=None,
                    assignments=parsed,
                    start_line=index + 1,
                    end_line=index + 1,
                    source_lines=(raw,),
                )
            )
        index += 1
    return (
        DslInitialConditionAdmission(
            declarations=tuple(declarations),
            materialized_source=_commented_initial_source(
                lines,
                claimed_indexes,
            ),
        ),
        frozenset(claimed_indexes),
    )


def _parse_dsl_ir(text: str, *, units: UnitsModel | None = None) -> "DSLIR":
    """
    Parse simulator DSL into a structured intermediate representation (IR).

    This is the single source of truth for top-level DSL admission. Immutable
    analysis and Mechanism construction derive from this IR so consumers never
    parse source independently.
    """
    u = units or UnitsModel()
    energy_unit = u.energy_unit
    temperature_K = float(u.temperature_K)
    standard_conc_M = float(u.standard_conc_M)
    kappa_global = 1.0

    from .state_model import StateNetwork, TSDegreeError
    from .computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START

    state_network = StateNetwork()
    state_facts: List[Dict[str, object]] = []
    edge_facts: List[Dict[str, object]] = []
    state_network_steps: List[Dict[str, object]] = []
    source_energy_unit_origin: Dict[str, object] | None = None
    steps: List[ParsedStep] = []
    algebra_lines: List[str] = []
    inline_initial_conditions, inline_initial_line_indexes = (
        _admit_inline_initial_conditions(text)
    )
    notes: List[str] = []

    # Parse temperature schedule if present
    from ..temperature_dsl import parse_temperature_schedule
    from ..intervention_schedule import _admit_intervention_directives

    temperature_schedule = parse_temperature_schedule(
        text,
        nominal_temperature=temperature_K,
    )
    intervention_schedule = _admit_intervention_directives(text)

    raw_lines = text.splitlines()
    if not any(line.strip() and not line.strip().startswith("#") for line in raw_lines):
        raise DSLError("DSL text is empty. Provide at least one reaction or directive.")
    source_origin_by_line = {
        origin.line_number: origin
        for origin in _source_origins(text)
    }

    lines = [ln.rstrip() for ln in raw_lines]
    in_generated_computational_mode_block = False
    generated_computational_mode_cm_ids: set[str] = set()
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        lower = line.lower()

        if line == GENERATED_BLOCK_START:
            in_generated_computational_mode_block = True
            continue
        if line == GENERATED_BLOCK_END:
            in_generated_computational_mode_block = False
            continue

        if not line or line.startswith("#"):
            continue

        # Computational Mode source-of-truth lines are embedded in the Reaction DSL
        # but are not part of the simulation DSL. Simulations consume only the
        # generated mechanism block, so `comp:` lines must be ignored here.
        if lower.startswith("comp:"):
            continue

        # Temperature schedule lines: parsed separately above
        if lower.startswith(("time:", "temp_step:", "temp_response:")):
            continue
        if lower.startswith("intervention:"):
            continue

        # Header-like switches
        if lower.startswith("energy="):
            try:
                energy_unit = _parse_energy_unit_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"energy unit set to {energy_unit}")
            source_origin = source_origin_by_line[line_no]
            source_energy_unit_origin = {
                "authored_ordinal": source_origin.authored_ordinal,
                "line_number": source_origin.line_number,
                "kind": source_origin.kind,
                "source_text": source_origin.source_text,
            }
            continue

        if lower.startswith("t="):
            raise DSLError(
                "Source-level T= is not supported; the mechanism tab "
                "temperature is the nominal scientific authority.",
                line_number=line_no,
                line_content=raw,
            )

        if lower.startswith("c0=") or lower.startswith("c°="):
            try:
                standard_conc_M = _parse_standard_conc_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"C° set to {standard_conc_M:g} M")
            continue

        if lower.startswith("κ=") or lower.startswith("kappa="):
            try:
                kappa_global = _parse_kappa_directive(line)
            except DSLError as exc:
                if exc.line_number is None:
                    raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                                   line_number=line_no, line_content=raw) from exc
                raise
            notes.append(f"κ set to {kappa_global:g}")
            continue

        if line_no - 1 in inline_initial_line_indexes:
            continue

        # State and edge definitions - parse into state network
        if lower.startswith("state:"):
            _, rest = line.split(":", 1)
            name, state_kwargs, authored_fields = _parse_state_step(
                rest,
                energy_unit=energy_unit,
                line_number=line_no,
                line_content=raw,
            )
            try:
                state_network.add_state(name, **state_kwargs)
            except ValueError as exc:
                raise DSLError(str(exc), line_number=line_no, line_content=raw) from exc
            source_origin = source_origin_by_line[line_no]
            origin = {
                "authored_ordinal": source_origin.authored_ordinal,
                "line_number": source_origin.line_number,
                "kind": source_origin.kind,
                "source_text": source_origin.source_text,
            }
            state_fact = {
                "name": name,
                "kind": state_kwargs["kind"],
                "energy_j_per_mol": state_kwargs["energy"][0],
                "source_energy_unit": authored_fields["energy_unit"],
                "source_energy_value": authored_fields["energy_value"],
                "degeneracy": state_kwargs["degeneracy"],
                "standard_state": state_kwargs["standard_state"],
                "members": state_kwargs["members"],
                "std_conc_product_M": state_kwargs["std_conc_product_M"],
                "authored_fields": authored_fields["fields"],
                "origin": origin,
            }
            state_facts.append(state_fact)
            state_network_steps.append(
                {
                    "step_ordinal": len(state_network_steps),
                    "kind": "state",
                    "state_index": len(state_facts) - 1,
                    "origin": origin,
                }
            )
            continue

        if lower.startswith("edge:"):
            _, rest = line.split(":", 1)
            a, b = _parse_edge_step(rest, line_number=line_no, line_content=raw)
            try:
                state_network.add_edge(a, b)
            except (KeyError, ValueError) as exc:
                raise DSLError(str(exc), line_number=line_no, line_content=raw) from exc
            source_origin = source_origin_by_line[line_no]
            origin = {
                "authored_ordinal": source_origin.authored_ordinal,
                "line_number": source_origin.line_number,
                "kind": source_origin.kind,
                "source_text": source_origin.source_text,
            }
            edge_fact = {
                "state_a": a,
                "state_b": b,
                "origin": origin,
            }
            edge_facts.append(edge_fact)
            state_network_steps.append(
                {
                    "step_ordinal": len(state_network_steps),
                    "kind": "edge",
                    "edge_index": len(edge_facts) - 1,
                    "origin": origin,
                }
            )
            continue

        classification = classify_parameter_algebra_declaration(raw, line_number=line_no)
        if classification.kind == "let":
            algebra_lines.append(raw)
            continue

        if classification.kind == "param":
            algebra_lines.append(raw)
            continue

        if classification.kind == "invalid_step_key_identifier":
            target = classification.raw_name or raw.strip()
            raise DSLError(
                f"{target!r} is a step-local DSL key and cannot be declared as a parameter or observable.",
                suggestion=(
                    "Use a canonical indexed mechanism parameter such as k1, kf1, kr1, or Keq1, "
                    "or choose a longer ordinary name."
                ),
                line_number=line_no,
                line_content=raw,
            )

        if classification.kind == "unsupported_bare_assignment":
            target = classification.raw_name or raw.strip()
            raise DSLError(
                f"Bare algebra assignment {target!r} is not supported.",
                suggestion="Use 'let name = expr' or 'param name = expr'.",
                line_number=line_no,
                line_content=raw,
            )

        # reaction: line
        if lower.startswith("reaction:"):
            if in_generated_computational_mode_block:
                raise DSLError(
                    "reaction rows are not accepted inside the generated Computational Mode block; "
                    "generated Computational Mode block rows must use explicit generated equilibrium lines.",
                    line_number=line_no,
                    line_content=raw,
                )
            _, rest = line.split(":", 1)
            steps.append(
                _parse_reaction_step(
                    rest,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    kappa_global=kappa_global,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        # equilibrium: line
        if lower.startswith("equilibrium:"):
            _, rest = line.split(":", 1)
            parsed_step = _parse_equilibrium_step(
                rest,
                energy_unit=energy_unit,
                T=temperature_K,
                C0=standard_conc_M,
                kappa_global=kappa_global,
                line_number=line_no,
                line_content=raw,
                generated_computational_mode=in_generated_computational_mode_block,
            )
            if bool(in_generated_computational_mode_block):
                cm_id = str(getattr(parsed_step, "cm_id", "") or "").strip()
                if cm_id in generated_computational_mode_cm_ids:
                    raise DSLError(
                        f"duplicate generated Computational Mode cm_id {cm_id!r}; generated fast equilibria must have unique cm_id values.",
                        line_number=line_no,
                        line_content=raw,
                    )
                generated_computational_mode_cm_ids.add(cm_id)
            steps.append(parsed_step)
            continue

        # Fallback: Try to parse as bare arrow syntax (no "reaction:" or "equilibrium:" prefix)
        if _ARROW_RE.search(line):
            if in_generated_computational_mode_block:
                raise DSLError(
                    "Bare arrow syntax is not accepted inside the generated Computational Mode block; "
                    "generated Computational Mode block rows must use explicit generated equilibrium lines.",
                    line_number=line_no,
                    line_content=raw,
                )
            steps.append(
                _parse_bare_arrow_step(
                    line,
                    energy_unit=energy_unit,
                    T=temperature_K,
                    C0=standard_conc_M,
                    kappa_global=kappa_global,
                    line_number=line_no,
                    line_content=raw,
                )
            )
            continue

        raise DSLError(f"unrecognized line: {line!r}", line_number=line_no, line_content=raw)

    if state_network.states() or state_network.edges():
        try:
            state_network.validate()
        except (TSDegreeError, ValueError) as exc:
            raise DSLError(str(exc)) from None

    if temperature_schedule is not None:
        notes.append(f"Temperature schedule detected: {temperature_schedule}")
    if intervention_schedule is not None:
        notes.append("Intervention schedule detected")

    return DSLIR(
        version=_DSL_IR_VERSION,
        energy_unit=str(energy_unit),
        temperature_K=float(temperature_K),
        standard_conc_M=float(standard_conc_M),
        kappa_global=float(kappa_global),
        temperature_schedule=temperature_schedule,
        intervention_schedule=intervention_schedule,
        state_network=state_network,
        state_network_facts={
            "source_energy_unit": str(energy_unit),
            "source_energy_unit_origin": source_energy_unit_origin,
            "states": tuple(state_facts),
            "edges": tuple(edge_facts),
            "steps": tuple(state_network_steps),
        },
        steps=steps,
        algebra_lines=algebra_lines,
        inline_initial_conditions=inline_initial_conditions,
        notes=notes,
    )


# ------------------------------ Mechanism building ---------------------------

@dataclass
class ParsedStep:
    """Internal representation of a parsed reaction step."""
    reactants: Dict[str, float]
    products: Dict[str, float]
    reversible: bool
    kf: float
    kr: Optional[float]
    model: str  # "Constant", "Arrhenius", or "Eyring"
    is_equilibrium: bool  # True if from "equilibrium:" line
    arrhenius_A: Optional[float] = None
    arrhenius_Ea_J_per_mol: Optional[float] = None
    eyring_dG_act_J_per_mol: Optional[float] = None
    kappa: Optional[float] = None
    standard_conc_M: Optional[float] = None
    dG_eq_J_per_mol: Optional[float] = None
    Keq_input: Optional[float] = None
    explicit_rates: List[float] = field(default_factory=list)
    user_kf_explicit: bool = False
    user_kr_explicit: bool = False
    cm_id: Optional[str] = None
    cm_std_ratio: Optional[float] = None
    generated_computational_mode: bool = False
    source_energy_unit: Optional[str] = None


def _parse_energy_unit_directive(line: str) -> str:
    from .common import normalize_energy_unit

    kv = _parse_keyvals(line)
    raw = kv.get("energy")
    if not raw:
        raise DSLError("energy= directive requires a value (kJ/mol, kcal/mol, or J/mol)")
    try:
        return normalize_energy_unit(raw)
    except ValueError:
        raise DSLError(
            f"energy must be 'kJ/mol', 'kcal/mol', or 'J/mol', got {raw!r}",
            examples=["energy=kJ/mol", "energy=kcal/mol", "energy=J/mol"],
        )


def _parse_standard_conc_directive(line: str) -> float:
    kv = _parse_keyvals(line)
    raw = kv.get("C0")
    if not raw:
        raise DSLError("C0= directive requires a numeric value")
    C0 = _float_or_none(raw)
    if C0 is None or C0 <= 0:
        raise DSLError("C0 must be positive")
    return float(C0)


def _parse_kappa_directive(line: str) -> float:
    kv = _parse_keyvals(line)
    raw = kv.get("κ")
    if not raw:
        raise DSLError("κ= directive requires a numeric value")
    kappa = _float_or_none(raw)
    if kappa is None or kappa <= 0:
        raise DSLError("κ must be positive")
    return float(kappa)


def _parse_state_step(
    rest: str,
    *,
    energy_unit: str,
    line_number: int,
    line_content: str,
) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    from .kinetics import normalize_energy_to_J_per_mol

    rest = rest.strip()
    try:
        name, kv = _parse_state_rest(rest)
    except (TypeError, ValueError) as exc:
        raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc

    if not name:
        raise DSLError(
            "state: requires a name",
            examples=["state: GS1 ; kind=GS ; energy=0.0"],
            line_number=line_number, line_content=line_content,
        )

    kind = (kv.get("kind") or "GS").strip().upper()
    energy_unit_val = kv.get("energy_unit") or energy_unit
    energy_val_raw = kv.get("energy")
    ej = 0.0
    if energy_val_raw is not None and str(energy_val_raw).strip() != "":
        tok = str(energy_val_raw).strip().split()
        if len(tok) == 2 and tok[1] in ("kJ/mol", "kcal/mol", "J/mol"):
            v0 = _float_or_none(tok[0])
            if v0 is None:
                raise DSLError(
                    "state energy must be numeric",
                    examples=["state: TS1 ; energy=85.5 kJ/mol"],
                    line_number=line_number, line_content=line_content,
                )
            ej = normalize_energy_to_J_per_mol(float(v0), tok[1])
        else:
            v0 = _float_or_none(tok[0])
            if v0 is None:
                raise DSLError(
                    "state energy must be numeric",
                    examples=["state: TS1 ; energy=85.5"],
                    line_number=line_number, line_content=line_content,
                )
            ej = normalize_energy_to_J_per_mol(float(v0), cast(str, energy_unit_val))

    degeneracy_raw = _float_or_none(kv.get("degeneracy"))
    degeneracy = 1.0 if degeneracy_raw is None else float(degeneracy_raw)
    standard_state = kv.get("standard_state") or "C0"
    members = None
    if kv.get("members") is not None and str(kv.get("members")).strip() != "":
        try:
            members = _parse_members_expr(str(kv.get("members")))
        except (TypeError, ValueError) as exc:
            raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc
    std_prod = None
    if kv.get("std") is not None and str(kv.get("std")).strip() != "":
        std_val = _extract_numeric_value(str(kv.get("std")))
        if std_val is None:
            raise DSLError("std must be numeric", line_number=line_number, line_content=line_content)
        std_prod = float(std_val)

    authored_energy_value = 0.0
    authored_energy_unit = str(energy_unit_val)
    if energy_val_raw is not None and str(energy_val_raw).strip():
        energy_tokens = str(energy_val_raw).strip().split()
        authored_energy_value = float(cast(float, _float_or_none(energy_tokens[0])))
        if len(energy_tokens) == 2:
            authored_energy_unit = energy_tokens[1]
    authored_fields = {
        str(key): str(value)
        for key, value in kv.items()
        if str(key) not in {"state", "name"}
    }
    return str(name), {
        "kind": kind,
        "energy": (ej, "J/mol"),
        "degeneracy": degeneracy,
        "standard_state": standard_state,
        "members": members,
        "std_conc_product_M": std_prod,
    }, {
        "energy_value": authored_energy_value,
        "energy_unit": authored_energy_unit,
        "fields": authored_fields,
    }


def _parse_edge_step(
    rest: str,
    *,
    line_number: Optional[int] = None,
    line_content: Optional[str] = None,
) -> Tuple[str, str]:
    rest = rest.strip()
    if "," in rest:
        parts = rest.split(",")
    elif "-" in rest:
        parts = rest.split("-")
    else:
        raise DSLError(f"edge must be 'A,B' or 'A-B', got: {rest}",
                       line_number=line_number, line_content=line_content)

    if len(parts) != 2:
        raise DSLError(f"edge must connect exactly 2 states, got: {rest}",
                       line_number=line_number, line_content=line_content)

    a = parts[0].strip()
    b = parts[1].strip()
    return a, b


def _split_stoich_and_params(
    text: str,
    *,
    reject_duplicate_canonical_keys: bool = False,
    line_number: int | None = None,
    line_content: str | None = None,
) -> Tuple[str, Dict[str, str]]:
    stoich_part, *tail = _SEMI_SPLIT_RE.split(text, maxsplit=1)
    params = (
        _parse_keyvals(
            tail[0],
            reject_duplicate_canonical_keys=reject_duplicate_canonical_keys,
            line_number=line_number,
            line_content=line_content,
        )
        if tail
        else {}
    )
    return stoich_part, params


# ----- shared validation helpers (C1-C3) used by both step parsers ---------

def _reject_per_step_globals(
    params: Dict[str, str],
    *,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    """Reject per-reaction T= and energy= overrides."""
    if "T" in params:
        raise DSLError(
            "Per-reaction T= is not supported; set the nominal temperature "
            "in the mechanism tab.",
            line_number=line_number, line_content=line_content,
        )
    if "energy" in params:
        raise DSLError(
            "Per-reaction energy= is not supported; use a global energy= directive",
            line_number=line_number, line_content=line_content,
        )


def _reject_unknown_params(
    params: Dict[str, str],
    known_keys: frozenset,
    step_label: str,
    *,
    line_number: Optional[int],
    line_content: Optional[str],
) -> None:
    """Reject parameter keys not in known_keys."""
    unknown = set(params) - known_keys
    if unknown:
        raise DSLError(
            f"Unknown {step_label} parameter(s): {', '.join(sorted(unknown))}",
            line_number=line_number, line_content=line_content,
        )


def _resolve_Keq_from_params(
    params: Dict[str, str],
    *,
    energy_unit: str,
    T: float,
    line_number: Optional[int],
    line_content: Optional[str],
) -> Tuple[float, Optional[float]]:
    """Resolve equilibrium constant Keq from explicit Keq input or dG_eq=.

    Precondition: at least one of "Keq" or "dG_eq" must be present in params.
    Returns (Keq, dG_eqJ_or_None). Raises DSLError on numeric/overflow/underflow issues.
    """
    from ..kinetics import K_from_deltaG_eq
    from .kinetics import normalize_energy_to_J_per_mol

    Keq = _float_or_none(params.get("Keq"))
    dG_eqJ: Optional[float] = None
    if Keq is None:
        dG_eq_val = _float_or_none(params["dG_eq"])
        if dG_eq_val is None:
            raise DSLError("dG_eq must be numeric", line_number=line_number, line_content=line_content)
        dG = normalize_energy_to_J_per_mol(dG_eq_val, energy_unit)
        dG_eqJ = dG
        try:
            Keq = K_from_deltaG_eq(dG, T)
        except OverflowError as exc:
            raise DSLError(
                "Equilibrium constant overflow (dG_eq too large for given T)",
                line_number=line_number, line_content=line_content,
            ) from exc
        if Keq <= 0:
            raise DSLError(
                "Equilibrium constant underflowed to zero (dG_eq too large for given T)",
                line_number=line_number, line_content=line_content,
            )
    return Keq, dG_eqJ


def _parse_reaction_like_step(
    *,
    stoich_part: str,
    params: Dict[str, str],
    energy_unit: str,
    T: float,
    C0: float,
    kappa_global: float,
    missing_eyring_message: str,
    line_number: int | None,
    line_content: str | None,
) -> ParsedStep:
    from ..kinetics import arrhenius_rate, eyring_rate
    from .common import molecularity
    from .kinetics import normalize_energy_to_J_per_mol

    try:
        react, prod, arrow = _parse_stoich(stoich_part)
    except DSLError as exc:
        if exc.line_number is None:
            raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                           line_number=line_number, line_content=line_content) from exc
        raise
    reversible = arrow in ("<->", "<=>")

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _REACTION_KNOWN_KEYS, "reaction",
                           line_number=line_number, line_content=line_content)

    model = _forward_rate_family_from_params(
        params,
        line_number=line_number,
        line_content=line_content,
    )
    kappa_raw = _float_or_none(params.get("κ"))
    if kappa_raw is not None:
        kappa = kappa_raw
    else:
        kappa = kappa_global
    n = molecularity(react)

    forward_rate_raw = params.get("kf") if "kf" in params else params.get("k")
    kf = _float_or_none(forward_rate_raw)
    kr = _float_or_none(params.get("kr"))
    has_forward_authority = any(
        key in params for key in ("kf", "k", "A", "Ea", "dG_act")
    )
    if reversible:
        _validate_equilibrium_rate_authority_for_dsl(
            has_kf=has_forward_authority,
            has_kr=("kr" in params),
            has_Keq=("Keq" in params),
            has_dG_eq=("dG_eq" in params),
            line_number=line_number,
            line_content=line_content,
        )
    if "kf" in params or "k" in params:
        _validate_rate_or_K(kf, "kf" if "kf" in params else "k",
                            line_number=line_number, line_content=line_content)
    if params.get("kr"):
        _validate_rate_or_K(kr, "kr", line_number=line_number, line_content=line_content)
    if "Keq" in params:
        _Keq_early = _float_or_none(params["Keq"])
        if _Keq_early is not None and _Keq_early <= 0:
            raise DSLError(
                f"Keq must be positive, got {_Keq_early}",
                line_number=line_number, line_content=line_content,
            )
    arrhenius_A = None
    arrhenius_EaJ = None
    eyring_dGJ = None
    dG_eqJ_for_step = None
    Keq_input = None
    explicit_rates: List[float] = []

    if model == "Arrhenius":
        A = _float_or_none(params.get("A"))
        Ea = params.get("Ea")
        if A is None or Ea is None:
            raise DSLError(
                "Arrhenius requires A and Ea",
                examples=["A -> B ; A=1e13 ; Ea=50"],
                line_number=line_number, line_content=line_content,
            )
        if A <= 0:
            raise DSLError(
                f"Arrhenius pre-exponential factor A must be positive, got {A}",
                line_number=line_number, line_content=line_content,
            )
        Ea_val = _float_or_none(Ea)
        if Ea_val is None:
            raise DSLError("Ea must be numeric", line_number=line_number, line_content=line_content)
        EaJ = normalize_energy_to_J_per_mol(Ea_val, energy_unit)
        arrhenius_A = A
        arrhenius_EaJ = EaJ
        try:
            kf = arrhenius_rate(A, EaJ, T)
        except OverflowError as exc:
            raise DSLError(
                "Arrhenius rate computation overflowed (A and Ea produce an unrepresentable rate)",
                line_number=line_number, line_content=line_content,
            ) from exc
        explicit_rates.append(kf)
        if reversible and kr is None:
            if "Keq" in params or "dG_eq" in params:
                Keq, dG_eqJ_for_step = _resolve_Keq_from_params(
                    params, energy_unit=energy_unit, T=T,
                    line_number=line_number, line_content=line_content,
                )
                kr = float(effective_reverse_rate_from_keq(kf, Keq))
                Keq_input = Keq if "Keq" in params else None
            else:
                raise DSLError(
                    "reversible Arrhenius step needs exactly one of kr or Keq/dG_eq",
                    examples=["A <-> B ; A=1e10 ; Ea=50 ; Keq=2.0"],
                    line_number=line_number, line_content=line_content,
                )
    elif model == "Eyring":
        if kf is None:
            if "dG_act" not in params:
                raise DSLError(
                    missing_eyring_message,
                    examples=["A -> B ; dG_act=75.5", "A -> B ; k=1.5"],
                    line_number=line_number, line_content=line_content,
                )
            dG_act_val = _float_or_none(params["dG_act"])
            if dG_act_val is None:
                raise DSLError("dG_act must be numeric", line_number=line_number, line_content=line_content)
            dGJ = normalize_energy_to_J_per_mol(dG_act_val, energy_unit)
            eyring_dGJ = dGJ
            if kappa <= 0:
                raise DSLError(
                    f"Per-step kappa must be positive for Eyring rate computation, got {kappa}",
                    line_number=line_number, line_content=line_content,
                )
            try:
                kf = eyring_rate(dGJ, T, kappa=kappa, molecularity=n, standard_conc_M=C0)
            except OverflowError as exc:
                raise DSLError(
                    "Eyring rate computation overflowed (dG_act too large for given T)",
                    line_number=line_number, line_content=line_content,
                ) from exc
            explicit_rates.append(kf)
        if reversible and kr is None:
            if "Keq" in params or "dG_eq" in params:
                Keq, dG_eqJ_for_step = _resolve_Keq_from_params(
                    params, energy_unit=energy_unit, T=T,
                    line_number=line_number, line_content=line_content,
                )
                kr = float(effective_reverse_rate_from_keq(kf, Keq))
                Keq_input = Keq if "Keq" in params else None
    else:
        if kf is None:
            raise DSLError(
                "Constant reaction requires explicit k or kf",
                line_number=line_number,
                line_content=line_content,
            )
        explicit_rates.append(kf)
        if reversible and kr is None:
            Keq, dG_eqJ_for_step = _resolve_Keq_from_params(
                params,
                energy_unit=energy_unit,
                T=T,
                line_number=line_number,
                line_content=line_content,
            )
            kr = float(effective_reverse_rate_from_keq(kf, Keq))
            Keq_input = Keq if "Keq" in params else None

    if kr is not None:
        explicit_rates.append(kr)

    return ParsedStep(
        reactants=react,
        products=prod,
        reversible=reversible,
        kf=kf,
        kr=kr,
        model=model,
        is_equilibrium=False,
        arrhenius_A=arrhenius_A,
        arrhenius_Ea_J_per_mol=arrhenius_EaJ,
        eyring_dG_act_J_per_mol=eyring_dGJ,
        kappa=kappa,
        standard_conc_M=C0,
        dG_eq_J_per_mol=dG_eqJ_for_step,
        Keq_input=Keq_input,
        explicit_rates=explicit_rates,
        user_kf_explicit=any(
            key in params for key in ("kf", "k", "A", "Ea", "dG_act")
        ),
        user_kr_explicit=bool(params.get("kr")),
    )


def _parse_reaction_step(
    rest: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    kappa_global: float,
    line_number: int,
    line_content: str,
) -> ParsedStep:
    stoich_part, params = _split_stoich_and_params(
        rest,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
    )
    return _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        T=T,
        C0=C0,
        kappa_global=kappa_global,
        missing_eyring_message="Eyring step requires dG_act or explicit k",
        line_number=line_number,
        line_content=line_content,
    )


def _parse_bare_arrow_step(
    line: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    kappa_global: float,
    line_number: int,
    line_content: str,
) -> ParsedStep:
    stoich_part, params = _split_stoich_and_params(
        line,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
    )
    return _parse_reaction_like_step(
        stoich_part=stoich_part,
        params=params,
        energy_unit=energy_unit,
        T=T,
        C0=C0,
        kappa_global=kappa_global,
        missing_eyring_message="Bare arrow syntax requires k (or kf), or dG_act for Eyring",
        line_number=line_number,
        line_content=line_content,
    )


def _parse_equilibrium_step(
    rest: str,
    *,
    energy_unit: str,
    T: float,
    C0: float,
    kappa_global: float,
    line_number: int,
    line_content: str,
    generated_computational_mode: bool = False,
) -> ParsedStep:
    from .common import molecularity
    from .kinetics import normalize_energy_to_J_per_mol

    stoich_part, params = _split_stoich_and_params(
        rest,
        reject_duplicate_canonical_keys=True,
        line_number=line_number,
        line_content=line_content,
    )
    try:
        react, prod, arrow = _parse_stoich(stoich_part)
    except DSLError as exc:
        if exc.line_number is None:
            raise DSLError(str(exc.message), suggestion=exc.suggestion, examples=exc.examples,
                           line_number=line_number, line_content=line_content) from exc
        raise
    if arrow not in ("<->", "<=>"):
        raise DSLError(
            "equilibrium must use '<->' or '<=>'",
            examples=["equilibrium: A <-> B ; Keq=2.0 ; kf=10"],
            line_number=line_number, line_content=line_content,
        )
    _ = molecularity(react)

    _reject_per_step_globals(params, line_number=line_number, line_content=line_content)
    _reject_unknown_params(params, _EQUILIBRIUM_KNOWN_KEYS, "equilibrium",
                           line_number=line_number, line_content=line_content)

    has_explicit_Keq = "Keq" in params
    has_explicit_dG_eq = "dG_eq" in params
    Keq = _float_or_none(params.get("Keq"))
    if has_explicit_Keq and Keq is None:
        raise DSLError(
            "Keq must be numeric",
            line_number=line_number,
            line_content=line_content,
        )
    if has_explicit_Keq and Keq is not None:
        _validate_rate_or_K(Keq, "Keq", line_number=line_number, line_content=line_content)
    dG_eqJ = None
    has_cm_generated_identity = bool(str(params.get("cm_id") or "").strip())
    has_cm_generated_authority = bool(generated_computational_mode)
    if (has_cm_generated_identity or params.get("cm_std_ratio") is not None) and not generated_computational_mode:
        raise DSLError(
            "Computational Mode generated equilibrium fields are only valid inside the generated Computational Mode block",
            line_number=line_number,
            line_content=line_content,
        )
    if has_cm_generated_authority and not has_cm_generated_identity:
        raise DSLError(
            "generated Computational Mode cm_id is required for generated fast equilibria.",
            line_number=line_number,
            line_content=line_content,
        )
    if has_cm_generated_authority and has_explicit_Keq:
        raise DSLError(
            "generated Computational Mode Keq is not accepted; generated fast equilibria must use kf with dG_eq, cm_id, and cm_std_ratio.",
            line_number=line_number,
            line_content=line_content,
        )
    if has_cm_generated_authority and not has_explicit_dG_eq:
        raise DSLError(
            "generated Computational Mode dG_eq is required for generated fast equilibria.",
            line_number=line_number,
            line_content=line_content,
        )

    exp_rates: List[float] = []
    kf_explicit = None
    kr_explicit = None
    if "kf" in params:
        kfv = _float_or_none(params["kf"])
        if kfv is None:
            raise DSLError("kf must be numeric", line_number=line_number, line_content=line_content)
        _validate_rate_or_K(kfv, "kf", line_number=line_number, line_content=line_content)
        kf_explicit = kfv
        exp_rates.append(kfv)
    if "kr" in params:
        krv = _float_or_none(params["kr"])
        if krv is None:
            raise DSLError("kr must be numeric", line_number=line_number, line_content=line_content)
        _validate_rate_or_K(krv, "kr", line_number=line_number, line_content=line_content)
        kr_explicit = krv
        if has_cm_generated_authority:
            raise DSLError(
                "generated Computational Mode kr is not accepted; generated fast equilibria must use kf with dG_eq and cm_std_ratio.",
                line_number=line_number,
                line_content=line_content,
            )
        exp_rates.append(krv)

    _validate_equilibrium_rate_authority_for_dsl(
        has_kf=kf_explicit is not None,
        has_kr=kr_explicit is not None,
        has_Keq=has_explicit_Keq,
        has_dG_eq=has_explicit_dG_eq,
        line_number=line_number,
        line_content=line_content,
    )

    if (
        Keq is None
        and not has_explicit_dG_eq
        and kf_explicit is not None
        and kr_explicit is not None
        and kr_explicit != 0
    ):
        Keq = kf_explicit / kr_explicit

    if Keq is None and has_explicit_dG_eq:
        dG_eq = (
            params["dG_eq"]
            if "dG_eq" in params
            else params.get("dg_eq")
        )
        dG_eq_val = _float_or_none(dG_eq)
        if dG_eq_val is None:
            raise DSLError("dG_eq must be numeric", line_number=line_number, line_content=line_content)
        dG_eqJ = normalize_energy_to_J_per_mol(dG_eq_val, energy_unit)
    cm_std_ratio = None
    if params.get("cm_std_ratio") is not None:
        cm_std_ratio = _float_or_none(params.get("cm_std_ratio"))
        if cm_std_ratio is None or cm_std_ratio <= 0.0:
            raise DSLError("cm_std_ratio must be positive and numeric", line_number=line_number, line_content=line_content)
    if has_cm_generated_authority and cm_std_ratio is None:
        raise DSLError(
            "generated Computational Mode cm_std_ratio is required for generated fast equilibria.",
            line_number=line_number,
            line_content=line_content,
        )

    fe = _derive_fast_equilibrium_rates_with_context(
        Keq=Keq,
        dG_eq_J_per_mol=dG_eqJ,
        T=T,
        explicit_rates=exp_rates or None,
        line_number=line_number,
        line_content=line_content,
    )
    kf = fe.kf if kf_explicit is None else kf_explicit
    kr = fe.kr if kr_explicit is None else kr_explicit
    if kr_explicit is None and (has_explicit_Keq or dG_eqJ is not None):
        reverse_std_ratio = (
            float(cm_std_ratio)
            if has_cm_generated_authority and cm_std_ratio is not None and dG_eqJ is not None
            else 1.0
        )
        kr = float(effective_reverse_rate_from_keq(float(kf), float(fe.Keq), reverse_std_ratio))

    return ParsedStep(
        reactants=react,
        products=prod,
        reversible=True,
        kf=kf,
        kr=kr,
        model="Eyring",
        is_equilibrium=True,
        eyring_dG_act_J_per_mol=None,
        kappa=kappa_global,
        standard_conc_M=C0,
        dG_eq_J_per_mol=dG_eqJ,
        Keq_input=(Keq if has_explicit_Keq else None),
        explicit_rates=exp_rates,
        user_kf_explicit=("kf" in params),
        user_kr_explicit=(("kr" in params) and not has_cm_generated_authority),
        cm_id=(str(params.get("cm_id") or "").strip() or None),
        cm_std_ratio=cm_std_ratio,
        generated_computational_mode=bool(generated_computational_mode),
        source_energy_unit=str(energy_unit),
    )


def executed_parameter_analysis_from_mechanism(
    mechanism: object,
    *,
    source_analysis: DslAnalysis,
) -> object:
    """Project execution parameters from an already prepared mechanism."""

    from kindred.core.simulation_result_payload import (
        ExecutedParameterAnalysis,
        ExecutedParameterValue,
    )
    from kindred.core.simulator.dsl_parameter_scan import (
        extract_parameters_from_mechanism,
    )

    if type(source_analysis) is not DslAnalysis:
        raise TypeError(
            "Executed analysis requires exact DslAnalysis evidence."
        )
    definitions = extract_parameters_from_mechanism(mechanism)
    values = tuple(
        ExecutedParameterValue(
            name=item.name,
            value=item.value,
            context=item.context,
            source=item.source,
            step_index=item.step_index,
            editable=item.editable,
            unit=item.unit,
            role=item.role,
            minimum=item.min,
            maximum=item.max,
        )
        for item in definitions
    )
    return ExecutedParameterAnalysis(
        source_analysis_identity=source_analysis.identity,
        parameters=values,
    )


def _source_origins(text: str) -> tuple[DslSourceOrigin, ...]:
    origins: list[DslSourceOrigin] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            kind = stripped.split(":", 1)[0].strip().lower()
        elif "=" in stripped:
            kind = stripped.split("=", 1)[0].strip().lower()
        else:
            kind = "statement"
        origins.append(
            DslSourceOrigin(
                authored_ordinal=len(origins),
                line_number=line_number,
                kind=kind or "statement",
                source_text=raw_line,
            )
        )
    return tuple(origins)


def _mechanism_build_input(
    ir: DSLIR,
    *,
    initials: Mapping[str, float] | None,
    intervention_schedule: object,
):
    """Freeze the private parse accumulator into the sole builder input."""

    from kindred.core.intervention_schedule import InterventionSchedule
    from kindred.core.simulator.dsl_build_input import (
        DslBuildEdge,
        DslBuildState,
        DslBuildStep,
        DslMechanismBuildInput,
    )
    from kindred.core.temperature import TemperatureSchedule

    if type(ir) is not DSLIR:
        raise TypeError("DSL builder freeze requires the exact private DSLIR.")
    if (
        intervention_schedule is not None
        and type(intervention_schedule) is not InterventionSchedule
    ):
        raise TypeError(
            "DSL builder freeze requires exact InterventionSchedule."
        )

    canonical_initials = {
        str(name): float(value)
        for name, value in dict(initials or {}).items()
    }

    schedule_json: str | None = None
    if ir.temperature_schedule is not None:
        if type(ir.temperature_schedule) is not TemperatureSchedule:
            raise TypeError(
                "DSL builder freeze requires exact TemperatureSchedule."
            )
        schedule_json = json.dumps(
            ir.temperature_schedule.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    return DslMechanismBuildInput(
        version=1,
        energy_unit=str(ir.energy_unit),
        temperature_K=float(ir.temperature_K),
        standard_conc_M=float(ir.standard_conc_M),
        kappa_global=float(ir.kappa_global),
        temperature_schedule_json=schedule_json,
        intervention_schedule=intervention_schedule,
        states=tuple(
            DslBuildState(
                name=str(state.name),
                kind=str(state.kind),
                energy_jmol=float(state.energy_jmol),
                degeneracy=float(state.degeneracy),
                standard_state=str(state.standard_state),
                members=(
                    tuple(str(member) for member in state.members)
                    if state.members is not None
                    else None
                ),
                std_conc_product_M=(
                    float(state.std_conc_product_M)
                    if state.std_conc_product_M is not None
                    else None
                ),
            )
            for state in ir.state_network.states()
        ),
        edges=tuple(
            DslBuildEdge(
                state_a=str(edge.a),
                state_b=str(edge.b),
            )
            for edge in ir.state_network.edges()
        ),
        steps=tuple(
            DslBuildStep(
                reactants=tuple(
                    (str(name), float(value))
                    for name, value in step.reactants.items()
                ),
                products=tuple(
                    (str(name), float(value))
                    for name, value in step.products.items()
                ),
                reversible=bool(step.reversible),
                kf=float(step.kf),
                kr=float(step.kr) if step.kr is not None else None,
                model=str(step.model),
                is_equilibrium=bool(step.is_equilibrium),
                arrhenius_A=(
                    float(step.arrhenius_A)
                    if step.arrhenius_A is not None
                    else None
                ),
                arrhenius_Ea_J_per_mol=(
                    float(step.arrhenius_Ea_J_per_mol)
                    if step.arrhenius_Ea_J_per_mol is not None
                    else None
                ),
                eyring_dG_act_J_per_mol=(
                    float(step.eyring_dG_act_J_per_mol)
                    if step.eyring_dG_act_J_per_mol is not None
                    else None
                ),
                kappa=(
                    float(step.kappa)
                    if step.kappa is not None
                    else None
                ),
                standard_conc_M=(
                    float(step.standard_conc_M)
                    if step.standard_conc_M is not None
                    else None
                ),
                dG_eq_J_per_mol=(
                    float(step.dG_eq_J_per_mol)
                    if step.dG_eq_J_per_mol is not None
                    else None
                ),
                Keq_input=(
                    float(step.Keq_input)
                    if step.Keq_input is not None
                    else None
                ),
                explicit_rates=tuple(
                    float(value) for value in step.explicit_rates
                ),
                user_kf_explicit=bool(step.user_kf_explicit),
                user_kr_explicit=bool(step.user_kr_explicit),
                cm_id=str(step.cm_id) if step.cm_id is not None else None,
                cm_std_ratio=(
                    float(step.cm_std_ratio)
                    if step.cm_std_ratio is not None
                    else None
                ),
                generated_computational_mode=bool(
                    step.generated_computational_mode
                ),
                source_energy_unit=(
                    str(step.source_energy_unit)
                    if step.source_energy_unit is not None
                    else None
                ),
            )
            for step in ir.steps
        ),
        algebra_lines=tuple(str(line) for line in ir.algebra_lines),
        initials=tuple(canonical_initials.items()),
    )


def prepare_dsl_request_set(
    text: str,
    *,
    initials: Mapping[str, float] | None = None,
    units: UnitsModel | None = None,
) -> PreparedDslRequestSet:
    """Admit source once and publish its one immutable typed analysis."""

    from kindred.core.intervention_schedule import (
        coerce_intervention_schedule,
    )
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulator.dsl_build import build_mechanism_from_ir
    from kindred.core.simulator.dsl_parameter_scan import (
        extract_parameters_from_mechanism,
    )
    from kindred.core.simulator.parameter_algebra import (
        mechanism_parameter_namespace,
        parse_parameter_algebra_spec_from_dsl_text,
    )

    payload = _dsl_identity_payload(text, units)
    normalized_text = str(payload["dsl"])
    ir = _parse_dsl_ir(text, units=units)
    intervention_schedule = coerce_intervention_schedule(
        ir.intervention_schedule
    )
    mechanism = build_mechanism_from_ir(
        _mechanism_build_input(
            ir,
            initials=initials,
            intervention_schedule=intervention_schedule,
        )
    )
    parameter_algebra_spec = parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_namespace=mechanism_parameter_namespace(mechanism),
    )
    definitions = tuple(
        DslParameterAnalysis(
            name=str(item.name),
            value=float(item.value),
            context=str(item.context),
            source=str(item.source),
            step_index=item.step_index,
            editable=bool(item.editable),
            unit=item.unit,
            role=item.role,
            minimum=item.min,
            maximum=item.max,
        )
        for item in extract_parameters_from_mechanism(mechanism)
    )
    species_names = tuple(str(item) for item in mechanism.species_names())
    identity = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    analysis = DslAnalysis(
        identity=f"dsl-analysis:{identity}",
        normalized_text=normalized_text,
        origins=_source_origins(text),
        species_names=species_names,
        observable_names=tuple(
            sorted(
                str(name)
                for name in parameter_algebra_spec.observable_names
                if str(name).strip()
            )
        ),
        parameters=definitions,
        inline_initial_conditions=ir.inline_initial_conditions,
        temperature_schedule=getattr(ir, "temperature_schedule", None),
        intervention_schedule=getattr(ir, "intervention_schedule", None),
        structure={
            "energy_unit": getattr(ir, "energy_unit", ""),
            "temperature_K": getattr(ir, "temperature_K", None),
            "standard_conc_M": getattr(ir, "standard_conc_M", None),
            "kappa_global": getattr(ir, "kappa_global", None),
            "step_count": len(tuple(getattr(ir, "steps", ()) or ())),
            "algebra_lines": tuple(
                getattr(ir, "algebra_lines", ()) or ()
            ),
            "state_network": {
                "analysis_identity": f"dsl-analysis:{identity}",
                **dict(getattr(ir, "state_network_facts", {}) or {}),
            },
            "mechanism_metadata": _canonical_dsl_value(
                dict(getattr(mechanism, "metadata", {}) or {})
            ),
            "intervention_declarative_fingerprint": (
                intervention_schedule.fingerprint
                if intervention_schedule is not None
                else ""
            ),
        },
    )
    metadata = getattr(mechanism, "metadata", None)
    if isinstance(metadata, dict):
        metadata["dsl_analysis_identity"] = analysis.identity
        metadata["dsl_source_origins"] = tuple(
            {
                "authored_ordinal": origin.authored_ordinal,
                "line_number": origin.line_number,
                "kind": origin.kind,
            }
            for origin in analysis.origins
        )
        metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE] = (
            intervention_schedule
        )
    return PreparedDslRequestSet(
        analysis=analysis,
        mechanism=mechanism,
        intervention_schedule=intervention_schedule,
        parameter_algebra_spec=parameter_algebra_spec,
    )


def _coordinate_source_parts(
    source_line: str,
) -> tuple[str, list[list[str]], str]:
    before_comment, marker, comment = source_line.partition("#")
    prefix, separator, parameter_text = before_comment.partition(";")
    tokens: list[list[str]] = []
    if separator:
        for raw_token in re.split(r"[;,]", parameter_text):
            token = raw_token.strip()
            if not token:
                continue
            key, equals, value = token.partition("=")
            tokens.append(
                [key.strip(), value.strip() if equals else ""]
            )
    return prefix.rstrip(), tokens, f"{marker}{comment}" if marker else ""


def _coordinate_source_line(
    prefix: str,
    tokens: list[list[str]],
    comment: str,
) -> str:
    body = prefix.rstrip()
    if tokens:
        body += " ; " + ", ".join(
            f"{key}={value}" if value else f"{key}="
            for key, value in tokens
        )
    if comment:
        body += f" {comment.strip()}"
    return body


def _coordinate_token_matches(candidate: str, target: str) -> bool:
    candidate_name = str(candidate).strip()
    target_name = str(target).strip()
    if target_name == "Keq":
        return candidate_name.casefold() == "keq"
    if candidate_name == "K":
        return False
    return candidate_name.casefold() == target_name.casefold()


def _set_coordinate_source_token(
    tokens: list[list[str]],
    *,
    token: str,
    value: float,
) -> None:
    rendered = f"{float(value):.15g}"
    matches = [
        index
        for index, pair in enumerate(tokens)
        if _coordinate_token_matches(pair[0], token)
    ]
    if matches:
        target = matches[0]
        tokens[target] = [token, rendered]
        for duplicate in reversed(matches[1:]):
            tokens.pop(duplicate)
        return
    tokens.append([token, rendered])


def _origin_line_positions(
    analysis: DslAnalysis,
) -> dict[int, int]:
    """Resolve immutable origin ordinals onto canonical source line indexes."""

    lines = analysis.normalized_text.splitlines()
    positions: dict[int, int] = {}
    cursor = 0
    for origin in analysis.origins:
        position = next(
            (
                index
                for index in range(cursor, len(lines))
                if lines[index] == origin.source_text
            ),
            None,
        )
        if position is None:
            raise ValueError(
                "Prepared DSL origin no longer matches canonical source."
            )
        positions[origin.authored_ordinal] = position
        cursor = position + 1
    return positions


def _authored_step_origin(
    analysis: DslAnalysis,
    *,
    step_index: int,
) -> DslSourceOrigin:
    generated = False
    authored_steps: list[DslSourceOrigin] = []
    for line in analysis.normalized_text.splitlines():
        stripped = line.strip()
        if stripped == "# --- BEGIN GENERATED COMPUTATIONAL MODE ---":
            generated = True
            continue
        if stripped == "# --- END GENERATED COMPUTATIONAL MODE ---":
            generated = False
            continue
        if generated:
            continue
        origin = next(
            (
                item
                for item in analysis.origins
                if item.source_text == line
                and item.authored_ordinal
                not in {
                    candidate.authored_ordinal
                    for candidate in authored_steps
                }
            ),
            None,
        )
        if origin is None:
            continue
        code = line.split("#", 1)[0].strip()
        if code.casefold().startswith("comp:"):
            continue
        if any(arrow in code for arrow in ("->", "=>", "<=>")):
            authored_steps.append(origin)
    if not 1 <= step_index <= len(authored_steps):
        raise LookupError(
            f"Mechanism step {step_index} has no exact source origin."
        )
    return authored_steps[step_index - 1]


def _scalar_origin(
    analysis: DslAnalysis,
    *,
    name: str,
) -> DslSourceOrigin:
    pattern = re.compile(
        rf"^\s*param\s+{re.escape(name)}\s*=",
        re.IGNORECASE,
    )
    matches = [
        origin
        for origin in analysis.origins
        if pattern.match(origin.source_text.split("#", 1)[0])
    ]
    if len(matches) != 1:
        raise LookupError(
            f"Scalar parameter {name!r} has no unique source origin."
        )
    return matches[0]


def _forward_model_payload(model: object) -> dict[str, object]:
    from kindred.core.kinetics import (
        ArrheniusRateModel,
        ConstantRateModel,
        EyringRateModel,
    )

    if isinstance(model, ConstantRateModel):
        return {"type": "Constant", "k": float(model.k)}
    if isinstance(model, ArrheniusRateModel):
        return {
            "type": "Arrhenius",
            "A": float(model.A),
            "Ea_J_per_mol": float(model.Ea_J_per_mol),
        }
    if isinstance(model, EyringRateModel):
        return {
            "type": "Eyring",
            "dG_act_J_per_mol": float(model.dG_act_J_per_mol),
            "kappa": float(model.kappa),
            "standard_conc_M": float(model.standard_conc_M),
        }
    raise TypeError("Coordinate update lost its typed rate family.")


def _reaction_model_overrides(model: object) -> dict[str, object]:
    payload = _forward_model_payload(model)
    family = str(payload.pop("type"))
    payload.pop("k", None)
    return {"model": family, **payload}


def derive_prepared_dsl_with_parameter_updates(
    base: PreparedDslRequestSet,
    updates: tuple[DslParameterCoordinateUpdate, ...],
    *,
    wegscheider_cyclicity_enabled: bool,
) -> PreparedDslRequestSet:
    """Derive typed Slider science without a text-input or reparse lane."""

    from kindred.core.equilibrium_rate_authority import (
        EquilibriumRateAuthorityKind,
        effective_equilibrium_keq,
        effective_reverse_rate_from_keq,
        normalize_existing_equilibrium_rate_authority,
    )
    from kindred.core.kinetics import (
        ArrheniusRateModel,
        EyringRateModel,
        forward_rate_at_temperature,
        forward_rate_model_for_equilibrium,
        forward_rate_model_for_reaction,
        write_reference_rate,
    )
    from kindred.core.mechanism_metadata import (
        EquilibriumMetadataKeys,
        MechanismMetadataKeys,
        MechanismMetadataView,
    )
    from kindred.core.simulator.dsl_parameter_scan import (
        extract_parameters_from_mechanism,
    )
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
    )
    from kindred.core.simulator.step_indexing import (
        lookup_step_param_target,
    )

    if type(base) is not PreparedDslRequestSet:
        raise TypeError("base must be exact PreparedDslRequestSet.")
    if base.analysis.inline_initial_conditions.has_declarations:
        raise ValueError(
            "Typed parameter derivation requires canonical materialized "
            "initial conditions."
        )
    if type(updates) is not tuple or any(
        type(item) is not DslParameterCoordinateUpdate
        for item in updates
    ):
        raise TypeError(
            "updates must be an exact tuple of "
            "DslParameterCoordinateUpdate values."
        )
    if type(wegscheider_cyclicity_enabled) is not bool:
        raise TypeError(
            "wegscheider_cyclicity_enabled must be an exact bool."
        )
    if not updates:
        raise ValueError("Typed DSL parameter derivation requires updates.")
    names = tuple(item.name for item in updates)
    if len(names) != len(set(names)):
        raise ValueError("DSL parameter update names must be unique.")

    mechanism = base.mechanism_for_execution()
    spec = base.parameter_algebra_spec_for_execution()
    metadata = getattr(mechanism, "metadata", None)
    if not isinstance(metadata, dict):
        raise TypeError("Prepared DSL mechanism metadata must be mutable.")
    metadata[
        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED
    ] = wegscheider_cyclicity_enabled
    metadata_view = MechanismMetadataView.from_metadata(metadata)
    temperature = float(base.analysis.structure["temperature_K"])
    if temperature != metadata_view.temperature_K:
        raise ValueError(
            "Prepared DSL nominal temperature metadata is inconsistent."
        )

    apply_parameter_algebra_spec_to_mechanism(
        spec,
        mechanism=mechanism,
        require_mutable=False,
    )
    parameter_by_name = {
        str(item.name): item
        for item in extract_parameters_from_mechanism(mechanism)
    }
    constrained = {
        str(name)
        for name in dict(metadata.get("constrained_params") or {})
    }
    source_lines = base.analysis.normalized_text.splitlines()
    origin_positions = _origin_line_positions(base.analysis)
    updated_origin_lines: dict[int, str] = {}
    scalar_values = metadata.get("scalar_params")
    if not isinstance(scalar_values, dict):
        scalar_values = {}
        metadata["scalar_params"] = scalar_values

    for update in updates:
        target = lookup_step_param_target(mechanism, update.name)
        definition = parameter_by_name.get(update.name)
        scalar_statement_known = any(
            statement.name == update.name
            for statement in spec.param_statements
        )
        if target is None and not scalar_statement_known:
            raise LookupError(
                f"Parameter {update.name!r} is absent from retained DSL."
            )
        if (
            target is not None
            and (
                definition is None
                or not bool(definition.editable)
            )
        ) or update.name in constrained:
            raise ValueError(
                f"Parameter {update.name!r} is not writable."
            )
        if target is None:
            scalar_info = dict(
                dict(metadata.get("scalar_param_info") or {}).get(
                    update.name
                )
                or {}
            )
            if (
                scalar_info.get("derived") is True
                or scalar_info.get("editable") is False
            ):
                raise ValueError(
                    f"Scalar parameter {update.name!r} is not writable."
                )
            statements = list(spec.param_statements)
            statement_index = next(
                (
                    index
                    for index, statement in enumerate(statements)
                    if statement.name == update.name
                ),
                None,
            )
            if statement_index is None:
                raise LookupError(
                    f"Scalar parameter {update.name!r} has no typed declaration."
                )
            origin = _scalar_origin(
                base.analysis,
                name=update.name,
            )
            old_line = source_lines[
                origin_positions[origin.authored_ordinal]
            ]
            before_comment, marker, comment = old_line.partition("#")
            match = re.match(
                rf"^(\s*param\s+{re.escape(update.name)}\s*=\s*)(.*?)\s*$",
                before_comment,
                flags=re.IGNORECASE,
            )
            if match is None:
                raise ValueError(
                    f"Scalar parameter {update.name!r} origin is invalid."
                )
            new_line = (
                f"{match.group(1)}{update.value:.15g}"
                + (f" {marker}{comment}".rstrip() if marker else "")
            )
            updated_origin_lines[origin.authored_ordinal] = new_line
            statement = statements[statement_index]
            statements[statement_index] = replace(
                statement,
                expr_src=f"{update.value:.15g}",
                line_content=new_line,
            )
            spec = replace(spec, param_statements=statements)
            scalar_values[update.name] = update.value
            continue

        kind, index, role, entry = target
        origin = _authored_step_origin(
            base.analysis,
            step_index=int(entry["step_index"]),
        )
        old_line = source_lines[
            origin_positions[origin.authored_ordinal]
        ]
        prefix, tokens, comment = _coordinate_source_parts(old_line)
        source_token = role
        source_value = update.value

        if kind == "reaction":
            reaction = mechanism.reactions[index]
            model = forward_rate_model_for_reaction(
                reaction,
                default_kappa=metadata_view.kappa_global,
                default_standard_conc_M=metadata_view.standard_conc_M,
            )
            rewritten = write_reference_rate(
                model,
                reference_rate=update.value,
                reference_temperature_K=temperature,
            )
            rate = forward_rate_at_temperature(rewritten, temperature)
            mechanism.reactions[index] = replace(
                reaction,
                rate=rate,
                overrides=_reaction_model_overrides(rewritten),
            )
            if isinstance(rewritten, ArrheniusRateModel):
                source_token = "Ea"
                source_value = UnitsModel(
                    energy_unit=metadata_view.energy_unit,
                ).from_jmol(rewritten.Ea_J_per_mol)
            elif isinstance(rewritten, EyringRateModel):
                source_token = "dG_act"
                source_value = UnitsModel(
                    energy_unit=metadata_view.energy_unit,
                ).from_jmol(rewritten.dG_act_J_per_mol)
            else:
                source_token = "k"
        elif kind == "equilibrium":
            equilibrium = mechanism.equilibria[index]
            equilibrium_metadata = dict(equilibrium.metadata)
            authority = normalize_existing_equilibrium_rate_authority(
                equilibrium
            )
            if role == "kf":
                model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=temperature,
                    default_kappa=metadata_view.kappa_global,
                    default_standard_conc_M=metadata_view.standard_conc_M,
                )
                rewritten = write_reference_rate(
                    model,
                    reference_rate=update.value,
                    reference_temperature_K=temperature,
                )
                kf = forward_rate_at_temperature(rewritten, temperature)
                equilibrium_metadata[
                    EquilibriumMetadataKeys.FORWARD_MODEL
                ] = _forward_model_payload(rewritten)
                kr = equilibrium.kr
                if authority.kind == EquilibriumRateAuthorityKind.KEQ:
                    keq = effective_equilibrium_keq(
                        equilibrium,
                        temperature_K=temperature,
                    )
                    if keq is None:
                        raise ValueError(
                            f"Parameter {update.name!r} lost Keq authority."
                        )
                    kr = effective_reverse_rate_from_keq(
                        kf,
                        keq,
                        authority.reverse_std_ratio,
                    )
                mechanism.equilibria[index] = replace(
                    equilibrium,
                    kf=kf,
                    kr=kr,
                    metadata=equilibrium_metadata,
                )
                source_energy_unit = str(
                    equilibrium_metadata.get(
                        EquilibriumMetadataKeys.SOURCE_ENERGY_UNIT
                    )
                    or metadata_view.energy_unit
                )
                if isinstance(rewritten, ArrheniusRateModel):
                    source_token = "Ea"
                    source_value = UnitsModel(
                        energy_unit=source_energy_unit,
                    ).from_jmol(rewritten.Ea_J_per_mol)
                elif isinstance(rewritten, EyringRateModel):
                    source_token = "dG_act"
                    source_value = UnitsModel(
                        energy_unit=source_energy_unit,
                    ).from_jmol(rewritten.dG_act_J_per_mol)
                else:
                    source_token = "kf"
            elif role == "kr":
                if authority.kind != EquilibriumRateAuthorityKind.KR:
                    raise ValueError(
                        f"Parameter {update.name!r} is derived, not writable."
                    )
                mechanism.equilibria[index] = replace(
                    equilibrium,
                    kr=update.value,
                    metadata=equilibrium_metadata,
                )
                source_token = "kr"
            elif role == "Keq":
                if not authority.has_explicit_keq_param:
                    raise ValueError(
                        f"Parameter {update.name!r} is derived, not writable."
                    )
                if update.value <= 0.0:
                    raise ValueError(
                        f"Parameter {update.name!r} must be positive."
                    )
                kf = float(equilibrium.kf)
                equilibrium_metadata[
                    EquilibriumMetadataKeys.KEQ_INPUT
                ] = update.value
                mechanism.equilibria[index] = replace(
                    equilibrium,
                    Keq=update.value,
                    kr=effective_reverse_rate_from_keq(
                        kf,
                        update.value,
                        authority.reverse_std_ratio,
                    ),
                    metadata=equilibrium_metadata,
                )
                source_token = "Keq"
            else:
                raise ValueError(
                    f"Unsupported equilibrium coordinate role {role!r}."
                )
        else:
            raise ValueError(
                f"Unsupported mechanism coordinate kind {kind!r}."
            )

        _set_coordinate_source_token(
            tokens,
            token=source_token,
            value=float(source_value),
        )
        updated_origin_lines[origin.authored_ordinal] = (
            _coordinate_source_line(prefix, tokens, comment)
        )

    for ordinal, line in updated_origin_lines.items():
        source_lines[origin_positions[ordinal]] = line
    updated_text = "\n".join(source_lines)
    updated_origins = tuple(
        DslSourceOrigin(
            authored_ordinal=origin.authored_ordinal,
            line_number=origin.line_number,
            kind=origin.kind,
            source_text=updated_origin_lines.get(
                origin.authored_ordinal,
                origin.source_text,
            ),
        )
        for origin in base.analysis.origins
    )

    algebra_lines = tuple(
        origin.source_text
        for origin in updated_origins
        if re.match(
            r"^\s*(?:param|let)\s+",
            origin.source_text.split("#", 1)[0],
            flags=re.IGNORECASE,
        )
    )
    metadata[MechanismMetadataKeys.ALGEBRA_TEXT] = "\n".join(algebra_lines)
    apply_parameter_algebra_spec_to_mechanism(
        spec,
        mechanism=mechanism,
        require_mutable=False,
    )
    derived_parameter_values = [
        DslParameterAnalysis(
            name=str(item.name),
            value=float(item.value),
            context=str(item.context),
            source=str(item.source),
            step_index=item.step_index,
            editable=bool(item.editable),
            unit=item.unit,
            role=item.role,
            minimum=item.min,
            maximum=item.max,
        )
        for item in extract_parameters_from_mechanism(mechanism)
    ]
    derived_scalar_info = dict(metadata.get("scalar_param_info") or {})
    for raw_name, raw_value in dict(
        metadata.get("scalar_params") or {}
    ).items():
        name = str(raw_name)
        if any(item.name == name for item in derived_parameter_values):
            continue
        info = dict(derived_scalar_info.get(name) or {})
        derived_parameter_values.append(
            DslParameterAnalysis(
                name=name,
                value=float(raw_value),
                context="Algebra scalar",
                source="Algebra parameter",
                step_index=None,
                editable=not bool(info.get("derived")),
                unit=str(info.get("unit") or "1"),
                role="scalar",
            )
        )
    definitions = tuple(derived_parameter_values)
    identity_hash = hashlib.sha256(
        json.dumps(
            {
                "base_analysis_identity": base.analysis.identity,
                "updates": tuple(
                    {"name": item.name, "value": item.value}
                    for item in updates
                ),
                "normalized_text": updated_text,
                "wegscheider_cyclicity_enabled": (
                    wegscheider_cyclicity_enabled
                ),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    identity = f"dsl-analysis-parameter-derived:{identity_hash}"
    structure = dict(
        cast(
            Mapping[str, object],
            _canonical_dsl_value(base.analysis.structure),
        )
    )
    structure["algebra_lines"] = algebra_lines
    analysis_metadata = dict(metadata)
    analysis_metadata.pop("parameter_algebra_spec", None)
    structure["mechanism_metadata"] = _canonical_dsl_value(
        analysis_metadata
    )
    structure["derived_from_analysis_identity"] = (
        base.analysis.identity
    )
    structure["parameter_updates"] = tuple(
        {"name": item.name, "value": item.value}
        for item in updates
    )
    state_network = dict(
        cast(Mapping[str, object], structure["state_network"])
    )
    state_network["analysis_identity"] = identity
    structure["state_network"] = state_network
    analysis = DslAnalysis(
        identity=identity,
        normalized_text=updated_text,
        origins=updated_origins,
        species_names=base.analysis.species_names,
        observable_names=base.analysis.observable_names,
        parameters=definitions,
        inline_initial_conditions=DslInitialConditionAdmission(
            materialized_source=updated_text
        ),
        temperature_schedule=base.analysis.temperature_schedule,
        intervention_schedule=base.analysis.intervention_schedule,
        structure=structure,
    )
    metadata["dsl_analysis_identity"] = identity
    metadata["dsl_source_origins"] = tuple(
        {
            "authored_ordinal": origin.authored_ordinal,
            "line_number": origin.line_number,
            "kind": origin.kind,
        }
        for origin in updated_origins
    )
    return PreparedDslRequestSet(
        analysis=analysis,
        mechanism=mechanism,
        intervention_schedule=(
            base.intervention_schedule_for_execution()
        ),
        parameter_algebra_spec=spec,
    )


def derive_prepared_dsl_with_observables(
    base: PreparedDslRequestSet,
    definitions: tuple[DslDerivedObservableDefinition, ...],
) -> PreparedDslRequestSet:
    """Derive one typed Fit-local observable authority from a retained base."""

    from kindred.core.mechanism_metadata import MechanismMetadataKeys

    if not isinstance(base, PreparedDslRequestSet):
        raise TypeError("base must be a PreparedDslRequestSet.")
    if base.analysis.inline_initial_conditions.has_declarations:
        raise ValueError(
            "Derived observables require canonical materialized initial "
            "conditions."
        )
    if type(definitions) is not tuple or any(
        type(item) is not DslDerivedObservableDefinition
        for item in definitions
    ):
        raise TypeError(
            "definitions must be an exact tuple of "
            "DslDerivedObservableDefinition values."
        )
    if not definitions:
        raise ValueError(
            "Derived observable authority requires at least one definition."
        )
    observable_names = tuple(item.name for item in definitions)
    if len(observable_names) != len(set(observable_names)):
        raise ValueError("Derived observable names must be unique.")
    scalars = tuple(
        scalar
        for definition in definitions
        for scalar in definition.scalar_parameters
    )
    scalar_names = tuple(item.name for item in scalars)
    if len(scalar_names) != len(set(scalar_names)):
        raise ValueError(
            "Derived observable scalar identities must be globally unique."
        )
    occupied = {
        *base.analysis.species_names,
        *base.analysis.observable_names,
        *(item.name for item in base.analysis.parameters),
    }
    collisions = sorted(
        (set(observable_names) | set(scalar_names)) & occupied
    )
    if collisions:
        raise ValueError(
            "Derived observable science conflicts with retained base name(s): "
            + ", ".join(repr(name) for name in collisions)
        )
    if set(observable_names) & set(scalar_names):
        raise ValueError(
            "Derived observable names and scalar names must be disjoint."
        )
    if any(
        "\n" in item.expression or "\r" in item.expression
        for item in definitions
    ):
        raise ValueError(
            "Derived observable expressions must be single-line Algebra "
            "expressions."
        )

    scalar_lines = tuple(
        f"param {item.name} = {item.value:.17g}"
        for item in scalars
    )
    observable_lines = tuple(
        f"let {item.name} = {item.expression}"
        for item in definitions
    )
    derived_lines = scalar_lines + observable_lines
    base_text = base.analysis.normalized_text.rstrip()
    derived_text = "\n".join((base_text, *derived_lines))
    units = UnitsModel(
        energy_unit=str(base.analysis.structure["energy_unit"]),
        temperature_K=float(base.analysis.structure["temperature_K"]),
        standard_conc_M=float(
            base.analysis.structure["standard_conc_M"]
        ),
    )
    admitted = prepare_dsl_request_set(
        derived_text,
        initials={},
        units=units,
    )

    def _bound_identity(value: float) -> float | str:
        if value == float("-inf"):
            return "-infinity"
        if value == float("inf"):
            return "infinity"
        return float(value)

    definition_payload = tuple(
        {
            "name": item.name,
            "expression": item.expression,
            "scalar_parameters": tuple(
                {
                    "name": scalar.name,
                    "value": scalar.value,
                    "minimum": _bound_identity(scalar.minimum),
                    "maximum": _bound_identity(scalar.maximum),
                    "scope": scalar.scope,
                }
                for scalar in item.scalar_parameters
            ),
        }
        for item in definitions
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "base_analysis_identity": base.analysis.identity,
                "definitions": definition_payload,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    derived_identity = f"dsl-analysis-derived:{identity}"
    first_line = len(base_text.splitlines()) + 1
    derived_origins = tuple(
        DslSourceOrigin(
            authored_ordinal=len(base.analysis.origins) + index,
            line_number=first_line + index,
            kind=(
                "derived-scalar"
                if index < len(scalar_lines)
                else "derived-observable"
            ),
            source_text=line,
        )
        for index, line in enumerate(derived_lines)
    )
    structure = dict(admitted.analysis.structure)
    state_network = dict(
        cast(Mapping[str, object], structure["state_network"])
    )
    state_network["analysis_identity"] = derived_identity
    structure["state_network"] = state_network
    structure["derived_from_analysis_identity"] = base.analysis.identity
    structure["derived_observable_definitions"] = definition_payload
    analysis = DslAnalysis(
        identity=derived_identity,
        normalized_text=admitted.analysis.normalized_text,
        origins=base.analysis.origins + derived_origins,
        species_names=admitted.analysis.species_names,
        observable_names=admitted.analysis.observable_names,
        parameters=admitted.analysis.parameters,
        inline_initial_conditions=admitted.analysis.inline_initial_conditions,
        temperature_schedule=admitted.analysis.temperature_schedule,
        intervention_schedule=admitted.analysis.intervention_schedule,
        structure=structure,
    )
    mechanism = admitted.mechanism_for_execution()
    metadata = getattr(mechanism, "metadata", None)
    if isinstance(metadata, dict):
        metadata["dsl_analysis_identity"] = analysis.identity
        metadata["dsl_source_origins"] = tuple(
            {
                "authored_ordinal": origin.authored_ordinal,
                "line_number": origin.line_number,
                "kind": origin.kind,
            }
            for origin in analysis.origins
        )
        metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE] = (
            admitted.intervention_schedule_for_execution()
        )
    return PreparedDslRequestSet(
        analysis=analysis,
        mechanism=mechanism,
        intervention_schedule=(
            admitted.intervention_schedule_for_execution()
        ),
        parameter_algebra_spec=(
            admitted.parameter_algebra_spec_for_execution()
        ),
    )
