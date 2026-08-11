"""Bounded validation for observables defined inside an open Fit session."""

from __future__ import annotations

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
    is_protected_step_key_identifier,
)
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
    is_protected_indexed_identifier,
)

__all__ = [
    "FitLocalObservable",
    "fit_observable_names",
    "validate_fit_local_observable",
]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class FitLocalObservable:
    name: str
    expression: str
    scalar_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        expression = str(self.expression or "").strip()
        raw_scalar_names = tuple(self.scalar_names)
        if any(not isinstance(item, str) for item in raw_scalar_names):
            raise TypeError("Fit-local observable scalar names must be strings.")
        scalar_names = tuple(item.strip() for item in raw_scalar_names)
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError("Fit-local observable name is not a valid identifier.")
        if not expression:
            raise ValueError("Fit-local observable expression must be nonblank.")
        if any(not _IDENTIFIER.fullmatch(item) for item in scalar_names):
            raise ValueError("Fit-local observable scalar names must be identifiers.")
        if len(scalar_names) != len(set(scalar_names)):
            raise ValueError("Fit-local observable scalar names must be unique.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "scalar_names", scalar_names)


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
        or is_protected_indexed_identifier(observable_name)
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
            is_protected_indexed_identifier(unknown)
            or is_protected_step_key_identifier(unknown)
        ):
            raise ValueError(
                f"Observable references protected identifier {unknown!r}."
            )
    canonical = canonicalize_observable_rhs_source(
        expression_source,
        mechanism_namespace=namespace,
    )
    return FitLocalObservable(
        observable_name,
        canonical,
        tuple(sorted(unknown_identifiers)),
    )
