from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Sequence, Set, Tuple

from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.errors import DSLError

_PARAM_STMT_RE = re.compile(r"^\s*param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", re.IGNORECASE)
_LET_STMT_RE = re.compile(r"^\s*let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_MECH_PARAM_RE = re.compile(r"^(k|kf|kr|Keq)(\d+)$")


@dataclass(frozen=True)
class ParameterAssignment:
    name: str
    expr_src: str
    line_number: int
    line_content: str


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
class ParameterAlgebraSpec:
    """
    Parsed `param` statements from an Algebra section.

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
    mechanism_param_names: Set[str]
    scalar_input_names: Set[str] = field(default_factory=set)

    def param_assignment_names(self) -> Set[str]:
        return {p.name for p in self.param_statements}

    def solver_param_names(self) -> Set[str]:
        return self.namespace_model().solver_param_names()

    def namespace_model(self) -> ParameterAlgebraNamespace:
        symtab = SymbolTable()
        return ParameterAlgebraNamespace(
            mechanism_param_names=set(self.mechanism_param_names),
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
    Return (line_number, raw_line) entries for lines inside the '# Algebra' section,
    excluding the '# Algebra' header itself.
    """
    lines = dsl_text.splitlines()
    out: List[Tuple[int, str]] = []
    in_algebra = False
    for ln, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.lower().startswith("# algebra"):
            in_algebra = True
            continue
        if stripped.lower().startswith("# ") and in_algebra and not stripped.lower().startswith("# algebra"):
            in_algebra = False
        if in_algebra:
            out.append((ln, raw.rstrip("\n")))
    return out


def extract_parameter_assignments_from_algebra_lines(
    algebra_lines: Sequence[Tuple[int, str]],
    *,
    mechanism_param_names: Set[str],
) -> List[ParameterAssignment]:
    assignments: List[ParameterAssignment] = []
    seen: Set[str] = set()

    for line_no, raw in algebra_lines:
        original = raw.rstrip("\n")
        stripped = original.strip()
        if not stripped or stripped.startswith("#"):
            continue

        code = strip_inline_comment(original).strip()
        if not code:
            continue

        m_param = _PARAM_STMT_RE.match(code)
        if m_param:
            name = m_param.group(1)
            expr = m_param.group(2).strip()
            if name in SymbolTable().protected_names() or name in SymbolTable().functions().keys():
                raise DSLError(
                    f"Invalid solver parameter name {name!r}",
                    suggestion="Choose a name that does not shadow protected constants/functions (e.g., 'a', 'Ea', 'scale').",
                    line_number=line_no,
                    line_content=original,
                )
            m_mech = _MECH_PARAM_RE.match(name)
            if m_mech and name not in mechanism_param_names:
                raise DSLError(
                    f"Unknown mechanism parameter {name!r} in Algebra param statement",
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

        m_let = _LET_STMT_RE.match(code)
        if m_let:
            target = m_let.group(1)
            if target in mechanism_param_names:
                raise DSLError(
                    f"{target!r} is a rate/equilibrium parameter; use 'param {target} = ...' for parameter algebra",
                    suggestion=f"Replace this with: param {target} = ...",
                    examples=[f"param {target} = 4*k2"],
                    line_number=line_no,
                    line_content=original,
                )
            continue

        m_assign = _ASSIGN_RE.match(code)
        if m_assign:
            target = m_assign.group(1)
            if target in mechanism_param_names:
                raise DSLError(
                    f"{target!r} is a rate/equilibrium parameter; use 'param {target} = ...' for parameter algebra",
                    suggestion=f"Replace this with: param {target} = ...",
                    examples=[f"param {target} = 4*k2"],
                    line_number=line_no,
                    line_content=original,
                )

    return assignments


def extract_parameter_assignments_from_dsl_text(
    dsl_text: str,
    *,
    mechanism_param_names: Set[str],
) -> List[ParameterAssignment]:
    return extract_parameter_assignments_from_algebra_lines(
        collect_algebra_section_lines(dsl_text),
        mechanism_param_names=mechanism_param_names,
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
        if code.lower().startswith("param "):
            continue
        m_let = _LET_STMT_RE.match(code)
        if m_let:
            out.add(m_let.group(1))
            continue
        m_assign = _ASSIGN_RE.match(code)
        if m_assign:
            out.add(m_assign.group(1))
    return out


def parse_parameter_algebra_spec_from_dsl_text(
    dsl_text: str,
    *,
    mechanism_param_names: Set[str],
    scalar_input_names: Set[str] | None = None,
) -> ParameterAlgebraSpec:
    lines = collect_algebra_section_lines(dsl_text)
    assignments = extract_parameter_assignments_from_algebra_lines(lines, mechanism_param_names=mechanism_param_names)
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
        mechanism_param_names=set(mechanism_param_names),
        scalar_input_names=set(scalar_input_names or ()),
    )
