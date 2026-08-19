"""Bounded validation for observables defined inside an open Fit session."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

from kindred.core.algebra.identifier_canonicalization import (
    canonicalize_algebra_block,
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
from kindred.core.algebra.simulation_series import (
    CompiledAlgebraSeries,
    evaluate_compiled_algebra_series_for_simulation,
)
from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.dsl_parse import DslAnalysis, PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra_spec import (
    is_protected_step_key_identifier,
)
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
    is_protected_indexed_identifier,
)

__all__ = [
    "AdmittedAlgebraProgram",
    "FitLocalObservable",
    "admit_fit_local_algebra",
    "apply_admitted_algebra",
    "fit_observable_names",
    "fit_scalar_names",
    "reenter_fit_local_algebra",
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


@dataclass(frozen=True, slots=True)
class AdmittedAlgebraProgram:
    """Immutable request-local Algebra admission bound to one canonical base.

    The source text and retained AST are the sole local-program authority.  A
    program is transient until :func:`apply_admitted_algebra` is called; Fit
    policy owns scalar values/scope and supplies those values only at Apply.
    """

    base_analysis: DslAnalysis
    source_text: str
    observables: tuple[FitLocalObservable, ...]
    scalar_names: tuple[str, ...]
    compiled: CompiledAlgebraSeries = field(repr=False)
    dependency_plan: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_analysis, DslAnalysis):
            raise TypeError("Algebra admission requires the complete base analysis.")
        if type(self.source_text) is not str or not self.source_text.strip():
            raise ValueError("Algebra admission requires exact local source text.")
        observables = tuple(self.observables)
        scalars = tuple(str(name) for name in self.scalar_names)
        if not observables or any(not isinstance(item, FitLocalObservable) for item in observables):
            raise TypeError("Algebra admission observables must be typed values.")
        if len({item.name for item in observables}) != len(observables):
            raise ValueError("Algebra admission observable names must be unique.")
        if len(set(scalars)) != len(scalars) or any(not _IDENTIFIER.fullmatch(name) for name in scalars):
            raise ValueError("Algebra admission scalar names must be unique identifiers.")
        if not isinstance(self.compiled, CompiledAlgebraSeries):
            raise TypeError("Algebra admission requires one retained compiled plan.")
        names = tuple(item.name for item in observables)
        if tuple(self.compiled.observable_names) != names:
            raise ValueError("Algebra admission AST names do not match observables.")
        dependencies = tuple(
            (str(name), tuple(str(dep) for dep in deps))
            for name, deps in self.dependency_plan
        )
        if tuple(name for name, _deps in dependencies) != names:
            raise ValueError("Algebra admission dependency plan is incomplete.")
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "scalar_names", scalars)
        object.__setattr__(self, "dependency_plan", dependencies)

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.observables)

    def reenter(self, base: PreparedDslRequestSet) -> "AdmittedAlgebraProgram":
        return reenter_fit_local_algebra(self, base)

    def evaluate(self, mechanism: object, **kwargs: object) -> tuple[dict[str, object], dict[str, float]]:
        return evaluate_compiled_algebra_series_for_simulation(
            mechanism, self.compiled, **kwargs  # type: ignore[arg-type]
        )


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


def _call_names(expr: ExprNode) -> set[str]:
    if isinstance(expr, CallNode):
        return {str(expr.name)} | set().union(*(_call_names(arg) for arg in expr.args))
    if isinstance(expr, UnaryNode):
        return _call_names(expr.rhs)
    if isinstance(expr, BinaryNode):
        return _call_names(expr.lhs) | _call_names(expr.rhs)
    return set()


def _admit_local_block(
    *,
    base: PreparedDslRequestSet,
    source_text: str,
    observables: tuple[FitLocalObservable, ...],
    scalar_names: tuple[str, ...],
) -> AdmittedAlgebraProgram:
    """Admit local source against the canonical base namespace exactly once."""

    mechanism = base.accepted_mechanism_for_structure()
    namespace = build_namespace_from_mechanism(mechanism)
    base_spec = base.parameter_algebra_spec_for_execution()
    occupied = (
        set(base.analysis.species_names)
        | set(base.analysis.observable_names)
        | set(base.analysis.parameters[i].name for i in range(len(base.analysis.parameters)))
        | set(base_spec.param_assignment_names())
        | namespace.flat_names()
        | set(SymbolTable().protected_names())
        | set(SymbolTable().functions())
    )
    collisions = sorted(set(item.name for item in observables) & occupied)
    if collisions:
        raise ValueError(
            "Local Algebra declaration conflicts with canonical science: "
            + ", ".join(repr(name) for name in collisions)
        )
    parsed = parse_algebra(source_text)
    if tuple(str(stmt.name) for stmt in parsed.lines) != tuple(item.name for item in observables):
        raise ValueError("Local Algebra source names do not match its typed declarations.")
    canonical = canonicalize_algebra_block(parsed, mechanism_namespace=namespace)
    unknown_species = sorted(
        {
            species
            for statement in canonical.lines
            for species in _species_refs(statement.expr)
            if species not in set(base.analysis.species_names)
        }
    )
    if unknown_species:
        raise ValueError(
            "Local Algebra references unknown species: "
            + ", ".join(repr(name) for name in unknown_species)
        )
    known = (
        namespace.flat_names()
        | set(base.analysis.species_names)
        | set(base.analysis.observable_names)
        | set(base_spec.param_assignment_names())
        | set(base_spec.namespace_model().scalar_input_names)
        | set(SymbolTable().protected_names())
        | set(SymbolTable().functions())
        | set(item.name for item in observables)
        | set(scalar_names)
    )
    unknown_functions = sorted(
        {name for statement in canonical.lines for name in _call_names(statement.expr)}
        - set(SymbolTable().functions())
    )
    if unknown_functions:
        raise ValueError(
            "Unknown Algebra function(s): " + ", ".join(repr(name) for name in unknown_functions)
        )
    unknown = sorted(
        {
            name
            for statement in canonical.lines
            for name in _identifiers(statement.expr)
            if name not in known
        }
    )
    if unknown:
        raise ValueError("Unresolved Algebra identifier(s): " + ", ".join(repr(name) for name in unknown))
    local_names = set(item.name for item in observables) | set(scalar_names)
    dependencies = tuple(
        (
            str(statement.name),
            tuple(sorted(_identifiers(statement.expr) & local_names)),
        )
        for statement in canonical.lines
    )
    compiled = CompiledAlgebraSeries(
        block=canonical,
        observable_names=tuple(item.name for item in observables),
    )
    return AdmittedAlgebraProgram(
        base_analysis=base.analysis,
        source_text=source_text,
        observables=observables,
        scalar_names=scalar_names,
        compiled=compiled,
        dependency_plan=dependencies,
    )


def admit_fit_local_algebra(
    *,
    base: PreparedDslRequestSet,
    observables: Iterable[FitLocalObservable],
    scalar_names: Iterable[str] = (),
) -> AdmittedAlgebraProgram:
    """Create one immutable request-local Algebra admission from declarations."""

    items = tuple(observables)
    if not items or any(not isinstance(item, FitLocalObservable) for item in items):
        raise TypeError("observables must contain FitLocalObservable values.")
    names = tuple(str(name).strip() for name in scalar_names)
    expected = {name for item in items for name in item.scalar_names}
    if set(names) != expected:
        raise ValueError("scalar_names must exactly cover observable scalar dependencies.")
    source = "# Algebra\n" + "\n".join(
        f"let {item.name} = {item.expression}" for item in items
    ) + "\n"
    return _admit_local_block(
        base=base,
        source_text=source,
        observables=items,
        scalar_names=names,
    )


def reenter_fit_local_algebra(
    admission: AdmittedAlgebraProgram,
    base: PreparedDslRequestSet,
) -> AdmittedAlgebraProgram:
    """Re-admit exact local source against a new base, failing closed on drift."""

    return _admit_local_block(
        base=base,
        source_text=admission.source_text,
        observables=admission.observables,
        scalar_names=admission.scalar_names,
    )


def apply_admitted_algebra(
    *,
    source: str,
    base: PreparedDslRequestSet,
    admission: AdmittedAlgebraProgram,
    scalar_values: Mapping[str, float] = {},
) -> tuple[str, PreparedDslRequestSet]:
    """Materialize local source and perform exactly one final canonical admission."""

    from kindred.core.mechanism_source import MechanismAuthoringSource, source_matches_analysis
    from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
    from kindred.core.units import UnitsModel

    if admission.base_analysis != base.analysis:
        raise ValueError("Apply admission is bound to a stale canonical base analysis.")
    if not source_matches_analysis(MechanismAuthoringSource.from_parts(reactions_text=source), base.analysis):
        raise ValueError("Apply source no longer matches the admitted canonical base.")
    values = {str(name): float(value) for name, value in dict(scalar_values).items()}
    if set(values) != set(admission.scalar_names):
        raise ValueError("Apply scalar values must exactly cover local scalar names.")
    scalar_lines = tuple(f"param {name} = {values[name]:.17g}" for name in admission.scalar_names)
    newline = "\r\n" if "\r\n" in source else "\n"
    sep = "" if source.endswith(("\n", "\r")) else newline
    rendered = source + sep + newline.join(scalar_lines + tuple(
        line for line in admission.source_text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ))
    metadata = getattr(base.accepted_mechanism_for_structure(), "metadata", {})
    standard_conc = float(getattr(metadata, "standard_conc_M", 1.0)) if not isinstance(metadata, Mapping) else float(metadata.get("standard_conc_M", 1.0))
    prepared = prepare_dsl_request_set(
        rendered,
        initials={},
        units=UnitsModel(
            energy_unit=base.analysis.energy_unit,
            temperature_K=base.analysis.temperature_K,
            standard_conc_M=standard_conc,
        ),
    )
    return rendered, prepared


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

    return tuple(sorted(str(name) for name in prepared_dsl.analysis.observable_names))


def fit_scalar_names(
    prepared_dsl: PreparedDslRequestSet,
) -> tuple[str, ...]:
    """Return retained canonical scalar assignments in the Fit namespace."""

    spec = prepared_dsl.parameter_algebra_spec_for_execution()
    return tuple(sorted(str(name) for name in spec.param_assignment_names()))


def validate_fit_local_observable(
    *,
    name: str,
    expression: str,
    prepared_dsl: PreparedDslRequestSet,
    additional_known_observables: tuple[str, ...] = (),
    additional_known_scalars: tuple[str, ...] = (),
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

    mechanism = prepared_dsl.accepted_mechanism_for_structure()
    namespace = build_namespace_from_mechanism(mechanism)
    species = set(prepared_dsl.analysis.species_names)
    retained_scalars = set(fit_scalar_names(prepared_dsl)) | {
        str(item) for item in additional_known_scalars
    }
    existing = set(prepared_dsl.analysis.observable_names) | {
        str(item) for item in additional_known_observables
    }
    symbols = SymbolTable()
    occupied_casefold = {
        item.casefold()
        for item in (
            species
            | existing
            | retained_scalars
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
    unknown_functions = sorted(
        _call_names(expr) - set(symbols.functions())
    )
    if unknown_functions:
        raise ValueError(
            "Unknown Algebra function(s): "
            + ", ".join(repr(name) for name in unknown_functions)
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
        | retained_scalars
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
    local_scalar_names = tuple(
        sorted(
            set(unknown_identifiers)
            | (_identifiers(expr) & {str(item) for item in additional_known_scalars})
        )
    )
    return FitLocalObservable(
        observable_name,
        canonical,
        local_scalar_names,
    )
