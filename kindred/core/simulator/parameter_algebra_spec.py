from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Sequence, Set, Tuple

from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import (
    MechanismParameterNamespace,
    is_protected_indexed_identifier,
)

_PARAM_STMT_RE = re.compile(r"^\s*param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", re.IGNORECASE)
_LET_STMT_RE = re.compile(r"^\s*let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_ARROW_RE = re.compile(r"<->|<=>|->|=>")
_MECH_PARAM_RE = re.compile(r"^(k|kf|kr|Keq)(\d+)$")
_MECH_PARAM_CI_RE = re.compile(r"^(?:k|kf|kr|keq)(\d+)$", re.IGNORECASE)
_PROTECTED_STEP_KEY_IDENTIFIERS = frozenset({"k", "kf", "kr", "keq"})
_NON_ALGEBRA_ASSIGNMENT_PREFIXES = {
    "comp",
    "edge",
    "energy",
    "equilibrium",
    "init",
    "initial",
    "kappa",
    "reaction",
    "state",
    "t",
    "temp_const",
    "temp_response",
    "temp_step",
    "time",
    "c0",
}


def _raise_invalid_protected_indexed_identifier_error(
    raw_name: str,
    *,
    suggested_names: Sequence[str],
    line_number: int,
    line_content: str,
) -> None:
    suggestions = ", ".join(str(name) for name in suggested_names)
    raise DSLError(
        f"{raw_name!r} is not a valid indexed parameter identifier.",
        suggestion=(
            "Use the canonical indexed parameter name"
            + (f" for this step: {suggestions}." if suggestions else ".")
            + " Exact protected indexed names must resolve to an existing mechanism parameter."
        ),
        line_number=line_number,
        line_content=line_content,
    )


def is_protected_indexed_parameter_identifier(raw_name: str) -> bool:
    return is_protected_indexed_identifier(raw_name)


def invalid_parameter_algebra_identifier_reference_message(
    raw_name: str,
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
    reject_unresolved_protected_indexed: bool = False,
) -> str | None:
    name = str(raw_name or "").strip()
    if not name:
        return None
    if is_protected_step_key_identifier(name):
        return (
            f"{name!r} is a step-local DSL key and cannot be used as an Algebra identifier. "
            "Use a canonical indexed mechanism parameter such as k1, kf1, kr1, or Keq1, "
            "or choose a longer ordinary name such as 'K1_test'."
        )
    if mechanism_namespace is not None:
        if mechanism_namespace.resolve(name).canonical_name is not None:
            return None
        invalid_protected = mechanism_namespace.invalid_protected_indexed_identifier(name)
        if invalid_protected is not None:
            suggestions = ", ".join(str(suggestion) for suggestion in invalid_protected.suggested_names)
            suggestion_clause = (
                f" Use canonical indexed parameter name(s) for this step: {suggestions}."
                if suggestions
                else " Use an existing canonical indexed mechanism parameter, or choose a longer ordinary name."
            )
            return f"{name!r} is not a valid indexed parameter identifier.{suggestion_clause}"
        return None
    if reject_unresolved_protected_indexed and is_protected_indexed_parameter_identifier(name):
        return (
            f"{name!r} is a protected indexed parameter identifier. "
            "Exact protected indexed names must resolve through the mechanism namespace."
        )
    return None


def is_protected_step_key_identifier(raw_name: str) -> bool:
    name = str(raw_name or "").strip()
    return name == "K" or name.lower() in _PROTECTED_STEP_KEY_IDENTIFIERS


def _raise_protected_observable_identifier_error(
    raw_name: str,
    *,
    line_number: int,
    line_content: str,
) -> None:
    raise DSLError(
        f"{raw_name!r} is a protected indexed parameter identifier and cannot be declared as an observable.",
        suggestion=(
            "Use 'param name = expr' for mechanism parameter algebra, or choose a longer observable name "
            "such as 'K1_test' if this is not a mechanism parameter."
        ),
        line_number=line_number,
        line_content=line_content,
    )


def _raise_protected_step_key_identifier_error(
    raw_name: str,
    *,
    line_number: int,
    line_content: str,
) -> None:
    raise DSLError(
        f"{raw_name!r} is a step-local DSL key and cannot be declared as a parameter or observable.",
        suggestion=(
            "Use a canonical indexed mechanism parameter such as k1, kf1, kr1, or Keq1, "
            "or choose a longer ordinary name such as 'K1_test'."
        ),
        line_number=line_number,
        line_content=line_content,
    )


def _raise_unsupported_bare_assignment_error(
    raw_name: str,
    *,
    line_number: int,
    line_content: str,
) -> None:
    raise DSLError(
        f"Bare algebra assignment {raw_name!r} is not supported.",
        suggestion="Use 'let name = expr' or 'param name = expr'.",
        line_number=line_number,
        line_content=line_content,
    )


@dataclass(frozen=True)
class ParameterAssignment:
    name: str
    expr_src: str
    line_number: int
    line_content: str


@dataclass(frozen=True)
class ParameterAlgebraDeclarationClassification:
    kind: str
    raw_name: str = ""
    code: str = ""
    line_number: int = 0
    line_content: str = ""
    canonical_name: str | None = None
    invalid_protected_indexed_identifier: bool = False
    invalid_protected_indexed_suggestions: Tuple[str, ...] = ()


def classify_parameter_algebra_declaration(
    line: str,
    *,
    line_number: int = 0,
    mechanism_namespace: MechanismParameterNamespace | None = None,
    allow_non_algebra_bare_assignment: bool = True,
) -> ParameterAlgebraDeclarationClassification:
    original = str(line or "").rstrip("\n")
    stripped = original.strip()
    if not stripped or stripped.startswith("#"):
        return ParameterAlgebraDeclarationClassification(
            kind="empty",
            line_number=int(line_number or 0),
            line_content=original,
        )
    code = strip_inline_comment(original).strip()
    if not code:
        return ParameterAlgebraDeclarationClassification(
            kind="empty",
            line_number=int(line_number or 0),
            line_content=original,
        )
    for kind, regex in (("param", _PARAM_STMT_RE), ("let", _LET_STMT_RE)):
        match = regex.match(code)
        if match is None:
            continue
        raw_name = str(match.group(1))
        rhs = str(match.group(2) or "").lstrip()
        if rhs.startswith("{"):
            return ParameterAlgebraDeclarationClassification(
                kind="non_algebra",
                code=code,
                line_number=int(line_number or 0),
                line_content=original,
            )
        if is_protected_step_key_identifier(raw_name):
            return ParameterAlgebraDeclarationClassification(
                kind="invalid_step_key_identifier",
                raw_name=raw_name,
                code=code,
                line_number=int(line_number or 0),
                line_content=original,
            )
        canonical_name = None
        invalid_protected_found = False
        invalid_suggestions: Tuple[str, ...] = ()
        if mechanism_namespace is not None:
            resolution = mechanism_namespace.resolve(raw_name)
            canonical_name = resolution.canonical_name
            if canonical_name is None:
                invalid_protected = mechanism_namespace.invalid_protected_indexed_identifier(raw_name)
                if invalid_protected is not None:
                    invalid_protected_found = True
                    invalid_suggestions = tuple(str(name) for name in invalid_protected.suggested_names)
        return ParameterAlgebraDeclarationClassification(
            kind=kind,
            raw_name=raw_name,
            code=code,
            line_number=int(line_number or 0),
            line_content=original,
            canonical_name=canonical_name,
            invalid_protected_indexed_identifier=bool(invalid_protected_found),
            invalid_protected_indexed_suggestions=invalid_suggestions,
        )
    match = None if _ARROW_RE.search(code) else _ASSIGN_RE.match(code)
    if match is not None:
        rhs = code[match.end():].lstrip()
        raw_name = str(match.group(1))
        if (
            allow_non_algebra_bare_assignment
            and (rhs.startswith("{") or raw_name.lower() in _NON_ALGEBRA_ASSIGNMENT_PREFIXES)
        ):
            return ParameterAlgebraDeclarationClassification(
                kind="non_algebra",
                code=code,
                line_number=int(line_number or 0),
                line_content=original,
            )
        return ParameterAlgebraDeclarationClassification(
            kind="unsupported_bare_assignment",
            raw_name=raw_name,
            code=code,
            line_number=int(line_number or 0),
            line_content=original,
        )
    return ParameterAlgebraDeclarationClassification(
        kind="non_algebra",
        code=code,
        line_number=int(line_number or 0),
        line_content=original,
    )


@dataclass(frozen=True)
class ParameterAlgebraNamespace:
    mechanism_param_names: Set[str]
    param_assignment_names: Set[str]
    scalar_input_names: Set[str]
    observable_names: Set[str]
    builtin_function_names: Set[str]
    protected_symbol_names: Set[str]

    def solver_param_names(self) -> Set[str]:
        return set(self.mechanism_param_names) | set(self.param_assignment_names)

    def reserved_identifier_names(self) -> Set[str]:
        return set(self.builtin_function_names) | set(self.protected_symbol_names)


@dataclass(frozen=True)
class ParameterOverrideWarning:
    param_name: str
    inline_name: str
    step_index: int
    message: str


@dataclass(frozen=True)
class ParameterAlgebraSpec:
    """
    Parsed `param` statements from mechanism DSL text.

    Semantics
    ---------
    - Scalar base parameters: `param a = 4` declares an adjustable solver parameter
      with a default value. It is NOT a constraint and may be overridden by sliders/fitting.
    - Derived parameters (constraints): any `param <name> = <expr>` that references
      other solver parameters OR targets a mechanism parameter (k1, k2, kf1, ...).
      Derived parameters are re-evaluated whenever solver parameters change.
    """

    param_statements: List[ParameterAssignment]
    observable_names: Set[str]
    mechanism_namespace: MechanismParameterNamespace
    scalar_input_names: Set[str] = field(default_factory=set)
    override_warnings: Tuple[ParameterOverrideWarning, ...] = field(default_factory=tuple)

    @property
    def mechanism_param_names(self) -> Set[str]:
        return self.mechanism_namespace.flat_names()

    def param_assignment_names(self) -> Set[str]:
        return {p.name for p in self.param_statements}

    def solver_param_names(self) -> Set[str]:
        return self.namespace_model().solver_param_names()

    def namespace_model(self) -> ParameterAlgebraNamespace:
        symtab = SymbolTable()
        return ParameterAlgebraNamespace(
            mechanism_param_names=self.mechanism_namespace.flat_names(),
            param_assignment_names=self.param_assignment_names(),
            scalar_input_names=set(self.scalar_input_names),
            observable_names=set(self.observable_names),
            builtin_function_names=set(symtab.functions().keys()),
            protected_symbol_names=set(symtab.protected_names()),
        )


def mechanism_parameter_name_pattern() -> re.Pattern[str]:
    return _MECH_PARAM_RE


def strip_inline_comment(line: str) -> str:
    """
    Strip inline comments using the same conservative heuristic as the GUI:
    remove the first '#' that is not inside [...]. (Algebra has no strings.)
    """
    if "#" not in line:
        return line
    bracket_depth = 0
    for i, ch in enumerate(line):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "#" and bracket_depth == 0:
            return line[:i].rstrip()
    return line


def collect_algebra_section_lines(dsl_text: str) -> List[Tuple[int, str]]:
    """
    Return (line_number, raw_line) entries for algebra lines anywhere in the DSL text.
    """
    out: List[Tuple[int, str]] = []
    for ln, raw in enumerate(dsl_text.splitlines(), start=1):
        stripped = raw.strip()
        classification = classify_parameter_algebra_declaration(raw, line_number=ln)
        if stripped and not stripped.startswith("#") and (
            classification.kind in {
                "param",
                "let",
                "invalid_step_key_identifier",
                "unsupported_bare_assignment",
            }
        ):
            out.append((ln, raw.rstrip("\n")))
    return out


def extract_parameter_assignments_from_algebra_lines(
    algebra_lines: Sequence[Tuple[int, str]],
    *,
    mechanism_namespace: MechanismParameterNamespace,
) -> List[ParameterAssignment]:
    assignments: List[ParameterAssignment] = []
    seen: Set[str] = set()
    mechanism_param_names = mechanism_namespace.flat_names()

    for line_no, raw in algebra_lines:
        original = raw.rstrip("\n")
        stripped = original.strip()
        if not stripped or stripped.startswith("#"):
            continue

        classification = classify_parameter_algebra_declaration(
            original,
            line_number=line_no,
            mechanism_namespace=mechanism_namespace,
        )
        code = classification.code
        if not code:
            continue

        if classification.kind == "param":
            raw_name = classification.raw_name
            resolution = mechanism_namespace.resolve(raw_name)
            if classification.invalid_protected_indexed_identifier:
                _raise_invalid_protected_indexed_identifier_error(
                    raw_name,
                    suggested_names=classification.invalid_protected_indexed_suggestions,
                    line_number=line_no,
                    line_content=original,
                )
            name = resolution.canonical_name or raw_name
            expr = _PARAM_STMT_RE.match(code).group(2).strip()  # type: ignore[union-attr]
            if name in SymbolTable().protected_names() or name in SymbolTable().functions().keys():
                raise DSLError(
                    f"Invalid solver parameter name {name!r}",
                    suggestion="Choose a name that does not shadow protected constants/functions (e.g., 'a', 'Ea', 'scale').",
                    line_number=line_no,
                    line_content=original,
                )
            m_mech = _MECH_PARAM_CI_RE.match(raw_name)
            if m_mech and resolution.canonical_name is None and name not in mechanism_param_names:
                raise DSLError(
                    f"Unknown mechanism parameter {raw_name!r} in Algebra param statement",
                    suggestion="Use an existing mechanism parameter (e.g., k1, k2, kf1, kr1, Keq1) or define the parameter on a reaction line.",
                    examples=["reaction: A -> B; k=1.0", "param k1 = 4*k2"],
                    line_number=line_no,
                    line_content=original,
                )
            if name in seen:
                raise DSLError(
                    f"Duplicate derived parameter definition for {name!r}",
                    suggestion="Define each derived parameter only once.",
                    line_number=line_no,
                    line_content=original,
                )
            seen.add(name)
            assignments.append(
                ParameterAssignment(
                    name=name,
                    expr_src=expr,
                    line_number=line_no,
                    line_content=original,
                )
            )
            continue

        if classification.kind == "invalid_step_key_identifier":
            _raise_protected_step_key_identifier_error(
                classification.raw_name,
                line_number=line_no,
                line_content=original,
            )

        if classification.kind == "let":
            target_raw = classification.raw_name
            resolution = mechanism_namespace.resolve(target_raw)
            target = resolution.canonical_name or target_raw
            if resolution.canonical_name is not None and target in mechanism_param_names:
                raise DSLError(
                    f"{target_raw!r} is a rate/equilibrium parameter; use 'param {target_raw} = ...' for parameter algebra",
                    suggestion=f"Replace this with: param {target_raw} = ...",
                    examples=[f"param {target_raw} = 4*k2"],
                    line_number=line_no,
                    line_content=original,
                )
            if is_protected_indexed_parameter_identifier(target_raw):
                _raise_protected_observable_identifier_error(
                    target_raw,
                    line_number=line_no,
                    line_content=original,
                )
            continue

        if classification.kind == "unsupported_bare_assignment":
            _raise_unsupported_bare_assignment_error(
                classification.raw_name,
                line_number=line_no,
                line_content=original,
            )

    return assignments


def extract_parameter_assignments_from_dsl_text(
    dsl_text: str,
    *,
    mechanism_namespace: MechanismParameterNamespace,
) -> List[ParameterAssignment]:
    return extract_parameter_assignments_from_algebra_lines(
        collect_algebra_section_lines(dsl_text),
        mechanism_namespace=mechanism_namespace,
    )


def extract_observable_names_from_algebra_lines(algebra_lines: Sequence[Tuple[int, str]]) -> Set[str]:
    out: Set[str] = set()
    for _line_no, raw in algebra_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        code = strip_inline_comment(stripped).strip()
        if not code:
            continue
        classification = classify_parameter_algebra_declaration(raw, line_number=_line_no)
        if classification.kind == "param":
            continue
        if classification.kind == "let":
            if is_protected_indexed_parameter_identifier(classification.raw_name):
                _raise_protected_observable_identifier_error(
                    classification.raw_name,
                    line_number=_line_no,
                    line_content=raw.rstrip("\n"),
                )
            if is_protected_step_key_identifier(classification.raw_name):
                _raise_protected_step_key_identifier_error(
                    classification.raw_name,
                    line_number=_line_no,
                    line_content=raw.rstrip("\n"),
                )
            out.add(classification.raw_name)
            continue
        if classification.kind == "invalid_step_key_identifier":
            _raise_protected_step_key_identifier_error(
                classification.raw_name,
                line_number=_line_no,
                line_content=raw.rstrip("\n"),
            )
        if classification.kind == "unsupported_bare_assignment":
            _raise_unsupported_bare_assignment_error(
                classification.raw_name,
                line_number=_line_no,
                line_content=raw.rstrip("\n"),
            )
    return out


def parse_parameter_algebra_spec_from_dsl_text(
    dsl_text: str,
    *,
    mechanism_namespace: MechanismParameterNamespace,
    scalar_input_names: Set[str] | None = None,
) -> ParameterAlgebraSpec:
    lines = collect_algebra_section_lines(dsl_text)
    assignments = extract_parameter_assignments_from_algebra_lines(lines, mechanism_namespace=mechanism_namespace)
    observables = extract_observable_names_from_algebra_lines(lines)
    for assignment in assignments:
        if assignment.name in observables:
            raise DSLError(
                f"Name {assignment.name!r} is defined as both a solver parameter (param) and an observable (let)",
                suggestion=(
                    f"Use only 'param {assignment.name} = ...' for solver parameters; "
                    f"use 'let {assignment.name}_obs = ...' for observables."
                ),
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
    return ParameterAlgebraSpec(
        param_statements=list(assignments),
        observable_names=set(observables),
        mechanism_namespace=mechanism_namespace,
        scalar_input_names=set(scalar_input_names or ()),
    )
