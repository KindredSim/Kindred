from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

from kindred.core.algebra.parser import (
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.algebra.identifier_canonicalization import canonicalize_observable_rhs_source
from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.parameter_algebra_spec import (
    classify_parameter_algebra_declaration,
    invalid_parameter_algebra_identifier_reference_message,
    is_protected_indexed_parameter_identifier,
    is_protected_step_key_identifier,
)
from kindred.core.simulator.parameter_namespace import MechanismParameterNamespace

__all__ = [
    "ObservableExpressionAnalysis",
    "analyze_observable_expression",
    "extract_observables_from_algebra_text",
    "detect_unknown_scalar_identifiers",
]


@dataclass(frozen=True)
class ObservableExpressionAnalysis:
    identifiers: Set[str]
    species_refs: Set[str]
    has_time_ref: bool


def _iter_identifiers(expr: ExprNode) -> Iterable[str]:
    if isinstance(expr, IdentNode):
        yield str(expr.name)
        return
    if isinstance(expr, SpeciesRefNode):
        return
    if isinstance(expr, UnaryNode):
        yield from _iter_identifiers(expr.rhs)
        return
    if isinstance(expr, BinaryNode):
        yield from _iter_identifiers(expr.lhs)
        yield from _iter_identifiers(expr.rhs)
        return
    if isinstance(expr, CallNode):
        for arg in expr.args:
            yield from _iter_identifiers(arg)
        return


def _iter_species_refs(expr: ExprNode) -> Iterable[str]:
    if isinstance(expr, SpeciesRefNode):
        yield str(expr.name)
        return
    if isinstance(expr, IdentNode):
        return
    if isinstance(expr, UnaryNode):
        yield from _iter_species_refs(expr.rhs)
        return
    if isinstance(expr, BinaryNode):
        yield from _iter_species_refs(expr.lhs)
        yield from _iter_species_refs(expr.rhs)
        return
    if isinstance(expr, CallNode):
        for arg in expr.args:
            yield from _iter_species_refs(arg)
        return


def analyze_observable_expression(expr_src: str) -> ObservableExpressionAnalysis:
    """
    Parse an Algebra expression using the project Algebra parser and return structural info.

    Notes
    -----
    - This uses a synthetic `# Algebra` block so the same grammar/compiler is exercised
      as in mechanism compilation.
    - Identifiers are *bare* IDENT tokens (not bracket species refs).
    """
    src = f"# Algebra\nlet __expr__ = {str(expr_src).strip()}\n"
    block = parse_algebra(src)
    if not block.lines:
        return ObservableExpressionAnalysis(identifiers=set(), species_refs=set(), has_time_ref=False)
    expr = block.lines[0].expr
    return ObservableExpressionAnalysis(
        identifiers=set(_iter_identifiers(expr)),
        species_refs=set(_iter_species_refs(expr)),
        has_time_ref=bool(expr.has_time_ref()),
    )


def _reject_invalid_identifier_references(
    identifiers: Set[str],
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
    reject_unresolved_protected_indexed: bool = False,
) -> None:
    for name in sorted(str(identifier) for identifier in (identifiers or set()) if str(identifier).strip()):
        message = invalid_parameter_algebra_identifier_reference_message(
            name,
            mechanism_namespace=mechanism_namespace,
            reject_unresolved_protected_indexed=bool(reject_unresolved_protected_indexed),
        )
        if message is not None:
            raise ValueError(message)


def extract_observables_from_algebra_text(
    algebra_text: str,
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
) -> dict[str, str]:
    """
    Extract observable definitions from the persisted Algebra editor text.

    Supports:
      - `let name = expr`

    Excludes:
      - `param name = expr` (scalar params / parameter algebra)
      - comments and blank lines

    Returns
    -------
    dict
        Mapping of observable_name -> expression_source (RHS text).
    """
    out: dict[str, str] = {}

    for raw in str(algebra_text or "").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Strip inline comments (ignore '#' inside bracket refs like [A]_0).
        code = stripped
        if "#" in stripped:
            bracket_depth = 0
            for i, ch in enumerate(stripped):
                if ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth = max(0, bracket_depth - 1)
                elif ch == "#" and bracket_depth == 0:
                    code = stripped[:i].rstrip()
                    break
        code = code.strip()
        if not code or code.startswith("#"):
            continue
        classification = classify_parameter_algebra_declaration(
            code,
            mechanism_namespace=mechanism_namespace,
            allow_non_algebra_bare_assignment=False,
        )
        if classification.kind == "param":
            continue

        if classification.kind == "let":
            if is_protected_indexed_parameter_identifier(classification.raw_name):
                raise ValueError(
                    f"{classification.raw_name!r} is a protected indexed parameter identifier "
                    "and cannot be declared as an observable."
                )
            if is_protected_step_key_identifier(classification.raw_name):
                raise ValueError(
                    f"{classification.raw_name!r} is a step-local DSL key "
                    "and cannot be declared as an observable."
                )
            rest = code[4:].strip()
        elif classification.kind == "invalid_step_key_identifier":
            raise ValueError(
                f"{classification.raw_name!r} is a step-local DSL key "
                "and cannot be declared as an observable."
            )
        elif classification.kind == "unsupported_bare_assignment":
            raise ValueError(
                f"Bare algebra assignment {classification.raw_name!r} is not supported. "
                "Use 'let name = expr' or 'param name = expr'."
            )
        else:
            continue

        if "=" not in rest:
            continue
        lhs, rhs = rest.split("=", 1)
        name = lhs.strip()
        expr = rhs.strip()
        if not name or not expr:
            continue
        analysis = analyze_observable_expression(expr)
        _reject_invalid_identifier_references(
            analysis.identifiers,
            mechanism_namespace=mechanism_namespace,
            reject_unresolved_protected_indexed=mechanism_namespace is None,
        )
        if mechanism_namespace is not None:
            expr = canonicalize_observable_rhs_source(expr, mechanism_namespace=mechanism_namespace)
        out[str(name)] = str(expr)
    return out


def detect_unknown_scalar_identifiers(
    expr_src: str,
    *,
    observable_name: str,
    known_identifiers: Set[str],
    mechanism_species: Set[str],
    mechanism_namespace: MechanismParameterNamespace | None = None,
) -> Set[str]:
    """
    Return bare identifier names that likely represent missing scalar parameters.

    Excludes:
    - Bracket species refs (they are not identifiers here)
    - Protected constants and builtin function names (SymbolTable)
    - The observable's own name
    - Any names already known to the caller (fit parameters, existing observables, etc.)
    - Mechanism species names (species must be bracketed in observables)
    """
    analysis = analyze_observable_expression(expr_src)
    symtab = SymbolTable()
    blocked = set(known_identifiers or set())
    blocked |= set(mechanism_species or set())
    blocked.add(str(observable_name))
    blocked |= set(symtab.protected_names())
    blocked |= set(symtab.functions().keys())
    _reject_invalid_identifier_references(
        analysis.identifiers,
        mechanism_namespace=mechanism_namespace,
    )
    identifier_names: set[str] = set()
    for raw_name in analysis.identifiers or set():
        name = str(raw_name or "").strip()
        if not name:
            continue
        if mechanism_namespace is not None:
            resolution = mechanism_namespace.resolve(name)
            if resolution.canonical_name is not None:
                identifier_names.add(str(resolution.canonical_name))
                continue
        identifier_names.add(name)
    candidates = {nm for nm in identifier_names if nm and nm not in blocked}
    _reject_invalid_identifier_references(
        candidates,
        mechanism_namespace=mechanism_namespace,
        reject_unresolved_protected_indexed=mechanism_namespace is None,
    )
    return candidates
