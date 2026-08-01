"""Bounded validation for observables defined inside an open Fit session."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from kindred.core.algebra.identifier_canonicalization import (
    canonicalize_observable_rhs_source,
)
from kindred.core.algebra.parser import (
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra_spec import (
    is_protected_indexed_parameter_identifier,
    is_protected_step_key_identifier,
)
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
)

__all__ = [
    "FitLocalObservable",
    "FitLocalScalar",
    "fit_observable_names",
    "validate_fit_local_observable",
]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class FitLocalScalar:
    name: str
    value: float = 1.0
    minimum: float = float("-inf")
    maximum: float = float("inf")
    scope: str = "shared"

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        value = float(self.value)
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        scope = str(self.scope or "").strip().lower()
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError("Fit-local scalar name is not a valid identifier.")
        if (
            not math.isfinite(value)
            or math.isnan(minimum)
            or math.isnan(maximum)
            or minimum >= maximum
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"Fit-local scalar {name!r} requires finite value within "
                "ordered non-NaN bounds."
            )
        if scope not in {"shared", "per_dataset"}:
            raise ValueError(
                "Fit-local scalar scope must be Shared or Per-dataset."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class FitLocalObservable:
    name: str
    expression: str
    scalar_parameters: tuple[FitLocalScalar, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        expression = str(self.expression or "").strip()
        scalars = tuple(self.scalar_parameters)
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError("Fit-local observable name is not a valid identifier.")
        if not expression:
            raise ValueError("Fit-local observable expression must be nonblank.")
        if any(not isinstance(item, FitLocalScalar) for item in scalars):
            raise TypeError(
                "Fit-local observable scalars must be FitLocalScalar values."
            )
        scalar_names = [item.name for item in scalars]
        if len(scalar_names) != len(set(scalar_names)):
            raise ValueError("Fit-local observable scalar names must be unique.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "scalar_parameters", scalars)


def _identifiers(expr: ExprNode) -> set[str]:
    if isinstance(expr, IdentNode):
        return {str(expr.name)}
    if isinstance(expr, SpeciesRefNode):
        return set()
    if isinstance(expr, UnaryNode):
        return _identifiers(expr.rhs)
    if isinstance(expr, BinaryNode):
        return _identifiers(expr.lhs) | _identifiers(expr.rhs)
    if isinstance(expr, CallNode):
        found: set[str] = set()
        for argument in expr.args:
            found |= _identifiers(argument)
        return found
    return set()


def _species_refs(expr: ExprNode) -> set[str]:
    if isinstance(expr, SpeciesRefNode):
        return {str(expr.name)}
    if isinstance(expr, IdentNode):
        return set()
    if isinstance(expr, UnaryNode):
        return _species_refs(expr.rhs)
    if isinstance(expr, BinaryNode):
        return _species_refs(expr.lhs) | _species_refs(expr.rhs)
    if isinstance(expr, CallNode):
        found: set[str] = set()
        for argument in expr.args:
            found |= _species_refs(argument)
        return found
    return set()


def fit_observable_names(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[str, ...]:
    """Return named algebra series available to the Fit evaluator."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    return tuple(sorted(str(name) for name in prepared_dsl.analysis.observable_names))


def validate_fit_local_observable(
    *,
    name: str,
    expression: str,
    prepared_dsl: PreparedDslRequestSet,
    additional_known_observables: tuple[str, ...] = (),
    unknown_scalar_scope: str = "shared",
) -> FitLocalObservable:
    """Validate one local definition without mutating canonical project text."""
    observable_name = str(name or "").strip()
    expression_source = str(expression or "").strip()
    if not _IDENTIFIER.fullmatch(observable_name):
        raise ValueError(
            "Observable name must match [A-Za-z_][A-Za-z0-9_]*."
        )
    if not expression_source:
        raise ValueError("Observable expression must be nonblank.")

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    mechanism = prepared_dsl.mechanism_for_execution()
    namespace = build_namespace_from_mechanism(mechanism)
    species = set(prepared_dsl.analysis.species_names)
    existing = set(prepared_dsl.analysis.observable_names) | {
        str(item) for item in additional_known_observables
    }
    symbols = SymbolTable()
    occupied_casefold = {
        item.casefold()
        for item in (
            species
            | existing
            | namespace.flat_names()
            | set(symbols.protected_names())
            | set(symbols.functions())
        )
    }
    if (
        observable_name.casefold() in occupied_casefold
        or is_protected_indexed_parameter_identifier(observable_name)
        or is_protected_step_key_identifier(observable_name)
    ):
        raise ValueError(
            f"Observable name {observable_name!r} conflicts with existing model science."
        )

    block = parse_algebra(
        f"# Algebra\nlet __kindred_fit_observable__ = {expression_source}\n"
    )
    if not block.lines:
        raise ValueError("Observable expression did not produce an Algebra value.")
    expr = block.lines[0].expr
    if expr.has_time_ref():
        raise ValueError(
            "Fit-local observables cannot use baseline/time-reference expressions."
        )
    unknown_species = _species_refs(expr) - species
    if unknown_species:
        raise ValueError(
            "Observable references unknown species: "
            + ", ".join(sorted(unknown_species))
        )
    bare_species = _identifiers(expr) & species
    if bare_species:
        raise ValueError(
            "Species must use bracket notation: "
            + ", ".join(sorted(bare_species))
        )
    if observable_name in _identifiers(expr):
        raise ValueError(
            f"Observable {observable_name!r} cannot reference itself."
        )
    allowed = (
        namespace.flat_names()
        | existing
        | set(symbols.protected_names())
        | set(symbols.functions())
    )
    unknown_identifiers = _identifiers(expr) - allowed
    for unknown in unknown_identifiers:
        if (
            is_protected_indexed_parameter_identifier(unknown)
            or is_protected_step_key_identifier(unknown)
        ):
            raise ValueError(
                f"Observable references protected identifier {unknown!r}."
            )
    scope = str(unknown_scalar_scope or "").strip().lower()
    if scope not in {"shared", "per_dataset"}:
        raise ValueError("Unknown scalar scope must be Shared or Per-dataset.")
    canonical = canonicalize_observable_rhs_source(
        expression_source,
        mechanism_namespace=namespace,
    )
    return FitLocalObservable(
        observable_name,
        canonical,
        tuple(
            FitLocalScalar(name=item, scope=scope)
            for item in sorted(unknown_identifiers)
        ),
    )
