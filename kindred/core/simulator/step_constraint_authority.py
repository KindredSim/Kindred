from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


_DEFAULT_TEMPERATURE_K = 298.15
_VALID_CONSTRAINT_REASONS = frozenset({"algebra", "wegscheider"})
_TOP_LEVEL_SCALAR_ASSIGNMENT_RE = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*[^#;\n]+\s*(?:#.*)?$")
_TOP_LEVEL_SCALAR_DIRECTIVES = frozenset({"energy", "t", "c0", "c°", "kappa", "κ"})
_ALGEBRA_STEP_PARAM_ASSIGNMENT_RE = re.compile(r"^\s*param\s+(?:k|kf|kr|K)\d+\s*=", re.IGNORECASE)
_MECHANISM_STEP_PARAM_RE = re.compile(r"^(k|kf|kr|K)(\d+)$")


@dataclass(frozen=True)
class StepConstraintAuthorityError:
    stage: str
    exc_type: str
    message: str


@dataclass(frozen=True)
class StepConstraintAuthorityAnalysis:
    constraint_reasons: Mapping[str, str]
    analysis_error: StepConstraintAuthorityError | None
    step_analysis_errors: Mapping[int, StepConstraintAuthorityError]
    mechanism_param_names: frozenset[str]
    scalar_input_names: frozenset[str]
    observable_names: frozenset[str]
    builtin_function_names: frozenset[str]
    protected_symbol_names: frozenset[str]
    referenced_scalar_inputs: frozenset[str]


@dataclass(frozen=True)
class _StepConstraintAuthorityState:
    analysis: StepConstraintAuthorityAnalysis
    mechanism: object | None


class StepConstraintAuthorityUnavailable(RuntimeError):
    """Raised when a target step cannot be semantically evaluated."""

    def __init__(self, *, step_index: int, analysis: StepConstraintAuthorityAnalysis):
        self.step_index = int(step_index)
        self.analysis = analysis
        step_error = analysis.step_analysis_errors.get(int(step_index))
        if step_error is not None:
            message = str(step_error.message)
        elif analysis.analysis_error is not None:
            message = str(analysis.analysis_error.message)
        else:
            message = f"Semantic authority is unavailable for step {int(step_index)}."
        super().__init__(message)


def _constraint_reason_from_entry(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    reason = entry.get("constraint_reason")
    if not isinstance(reason, str):
        return None
    normalized = reason.strip().lower()
    if normalized in _VALID_CONSTRAINT_REASONS:
        return normalized
    return None


def _context_temperature_K(context: Mapping[str, object] | None) -> float:
    if not isinstance(context, Mapping):
        return _DEFAULT_TEMPERATURE_K
    raw = context.get("temperature_K", _DEFAULT_TEMPERATURE_K)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TEMPERATURE_K


def _context_wegscheider_enabled(context: Mapping[str, object] | None) -> bool:
    if not isinstance(context, Mapping):
        return False
    return bool(context.get("wegscheider_cyclicity_enabled", False))


def _extract_top_level_scalar_assignments(source_text: str) -> tuple[str, dict[str, float]]:
    sanitized_lines: list[str] = []
    scalar_values: dict[str, float] = {}
    in_algebra_section = False

    for raw_line in str(source_text or "").splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()
        if lower.startswith("# algebra"):
            in_algebra_section = True
            sanitized_lines.append(raw_line)
            continue
        if lower.startswith("# ") and in_algebra_section and not lower.startswith("# algebra"):
            in_algebra_section = False
        if in_algebra_section or not stripped or stripped.startswith("#"):
            sanitized_lines.append(raw_line)
            continue
        match = _TOP_LEVEL_SCALAR_ASSIGNMENT_RE.match(raw_line)
        if match is None:
            sanitized_lines.append(raw_line)
            continue
        name, _, value_text = stripped.partition("=")
        normalized_name = str(name).strip()
        if normalized_name.lower() in _TOP_LEVEL_SCALAR_DIRECTIVES:
            sanitized_lines.append(raw_line)
            continue
        try:
            scalar_values[normalized_name] = float(value_text.split("#", 1)[0].strip())
        except (TypeError, ValueError):
            sanitized_lines.append(raw_line)
            continue

    return "\n".join(sanitized_lines), scalar_values


def _seed_scalar_params_into_mechanism(mechanism: object, scalar_values: Mapping[str, float]) -> None:
    if not scalar_values:
        return
    metadata = getattr(mechanism, "metadata", None)
    if not isinstance(metadata, dict):
        return
    scalar_params = metadata.get("scalar_params")
    if not isinstance(scalar_params, dict):
        scalar_params = {}
        metadata["scalar_params"] = scalar_params
    for raw_name, raw_value in scalar_values.items():
        scalar_params[str(raw_name)] = float(raw_value)


def _text_has_potential_step_constraints(source_text: str) -> bool:
    in_algebra_section = False
    for raw_line in str(source_text or "").splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()
        if lower.startswith("# algebra"):
            in_algebra_section = True
            continue
        if lower.startswith("# ") and in_algebra_section and not lower.startswith("# algebra"):
            in_algebra_section = False
        if not in_algebra_section or not stripped or stripped.startswith("#"):
            continue
        code = raw_line.split("#", 1)[0].strip()
        if _ALGEBRA_STEP_PARAM_ASSIGNMENT_RE.match(code):
            return True
    return False


def _build_step_constraint_authority_error(stage: str, exc: Exception) -> StepConstraintAuthorityError:
    return StepConstraintAuthorityError(
        stage=str(stage),
        exc_type=exc.__class__.__name__,
        message=str(exc),
    )


def _empty_step_constraint_authority_analysis(
    *,
    scalar_input_names: frozenset[str] | None = None,
    mechanism_param_names: frozenset[str] | None = None,
    observable_names: frozenset[str] | None = None,
    builtin_function_names: frozenset[str] | None = None,
    protected_symbol_names: frozenset[str] | None = None,
    referenced_scalar_inputs: frozenset[str] | None = None,
    analysis_error: StepConstraintAuthorityError | None = None,
    step_analysis_errors: Mapping[int, StepConstraintAuthorityError] | None = None,
) -> StepConstraintAuthorityAnalysis:
    return StepConstraintAuthorityAnalysis(
        constraint_reasons={},
        analysis_error=analysis_error,
        step_analysis_errors=dict(step_analysis_errors or {}),
        mechanism_param_names=frozenset(mechanism_param_names or ()),
        scalar_input_names=frozenset(scalar_input_names or ()),
        observable_names=frozenset(observable_names or ()),
        builtin_function_names=frozenset(builtin_function_names or ()),
        protected_symbol_names=frozenset(protected_symbol_names or ()),
        referenced_scalar_inputs=frozenset(referenced_scalar_inputs or ()),
    )


def _step_index_from_parameter_name(name: str) -> int | None:
    match = _MECHANISM_STEP_PARAM_RE.match(str(name))
    if match is None:
        return None
    return int(match.group(2))


def _assignment_name_from_dsl_error(assignments: tuple[object, ...], err: Exception) -> str | None:
    line_number = getattr(err, "line_number", None)
    if isinstance(line_number, int):
        for assignment in assignments:
            if int(getattr(assignment, "line_number", -1)) == int(line_number):
                return str(getattr(assignment, "name"))
    line_content = getattr(err, "line_content", None)
    if isinstance(line_content, str):
        stripped_line = line_content.strip()
        for assignment in assignments:
            if str(getattr(assignment, "line_content", "")).strip() == stripped_line:
                return str(getattr(assignment, "name"))
    return None


def _collect_affected_assignment_names(
    analyses_by_name: Mapping[str, object],
    direct_error_names: set[str],
) -> set[str]:
    reverse_dependencies: dict[str, set[str]] = {}
    for assignment_name, assignment_analysis in analyses_by_name.items():
        for dependency_name in getattr(assignment_analysis, "assignment_dependencies", frozenset()):
            reverse_dependencies.setdefault(str(dependency_name), set()).add(str(assignment_name))
    affected = set(str(name) for name in direct_error_names)
    stack = list(affected)
    while stack:
        current = stack.pop()
        for dependent_name in sorted(reverse_dependencies.get(current, ())):
            if dependent_name in affected:
                continue
            affected.add(dependent_name)
            stack.append(dependent_name)
    return affected


def _root_error_names_for_assignment(
    assignment_name: str,
    analyses_by_name: Mapping[str, object],
    direct_error_names: set[str],
) -> set[str]:
    roots: set[str] = set()
    stack = [str(assignment_name)]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in direct_error_names:
            roots.add(current)
            continue
        assignment_analysis = analyses_by_name.get(current)
        for dependency_name in getattr(assignment_analysis, "assignment_dependencies", frozenset()):
            stack.append(str(dependency_name))
    return roots


def _build_step_analysis_errors(
    analyses_by_name: Mapping[str, object],
    direct_errors_by_name: Mapping[str, StepConstraintAuthorityError],
    affected_assignment_names: set[str],
) -> dict[int, StepConstraintAuthorityError]:
    step_names_by_index: dict[int, set[str]] = {}
    for assignment_name in affected_assignment_names:
        step_index = _step_index_from_parameter_name(str(assignment_name))
        if step_index is None:
            continue
        step_names_by_index.setdefault(step_index, set()).add(str(assignment_name))

    step_errors: dict[int, StepConstraintAuthorityError] = {}
    direct_error_names = set(direct_errors_by_name)
    for step_index, step_names in step_names_by_index.items():
        direct_step_names = sorted(name for name in step_names if name in direct_error_names)
        if direct_step_names:
            step_errors[int(step_index)] = direct_errors_by_name[direct_step_names[0]]
            continue
        root_names = sorted(
            {
                root_name
                for step_name in step_names
                for root_name in _root_error_names_for_assignment(step_name, analyses_by_name, direct_error_names)
            }
        )
        if not root_names:
            continue
        exemplar = direct_errors_by_name[root_names[0]]
        step_errors[int(step_index)] = StepConstraintAuthorityError(
            stage=str(exemplar.stage),
            exc_type=str(exemplar.exc_type),
            message=f"Depends on invalid parameter algebra assignment(s): {', '.join(root_names)}",
        )
    return step_errors


def _build_step_constraint_authority_state_from_text(
    source_text: str,
    *,
    context: Mapping[str, object] | None = None,
) -> _StepConstraintAuthorityState:
    from kindred.core.algebra.symbols import SymbolTable
    from kindred.core.batch_initial_conditions import strip_named_reaction_dsl_initial_concentration_sets
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.parameter_algebra import (
        apply_parameter_algebra_spec_to_mechanism,
        mechanism_parameter_names,
        parse_parameter_algebra_spec_from_dsl_text,
        read_mechanism_parameter_values,
    )
    from kindred.core.simulator.parameter_algebra_eval import (
        analyze_parameter_algebra_assignments,
        build_parameter_algebra_evaluation_model,
    )
    from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
    from kindred.core.units import UnitsModel

    temperature_K = _context_temperature_K(context)
    wegscheider_enabled = _context_wegscheider_enabled(context)

    def _fresh_mechanism(sanitized_source_text: str, scalar_params: Mapping[str, float], *, wegscheider: bool):
        mechanism = parse_dsl_to_mechanism(
            sanitized_source_text,
            initials={},
            units=UnitsModel(temperature_K=temperature_K),
        )
        metadata = getattr(mechanism, "metadata", None)
        if isinstance(metadata, dict):
            metadata["wegscheider_cyclicity_enabled"] = wegscheider
        _seed_scalar_params_into_mechanism(mechanism, scalar_params)
        return mechanism

    def _state_from_scoped_failures(
        *,
        spec: ParameterAlgebraSpec,
        namespace,
        analyses_by_name: Mapping[str, object],
        affected_assignment_names: set[str],
        step_analysis_errors: Mapping[int, StepConstraintAuthorityError],
        scoped_apply_stage: str,
        sanitized_source_text: str,
        scalar_params: Mapping[str, float],
        wegscheider: bool,
    ) -> _StepConstraintAuthorityState:
        valid_assignments = [
            assignment for assignment in spec.param_statements if str(assignment.name) not in affected_assignment_names
        ]
        scoped_spec = ParameterAlgebraSpec(
            param_statements=list(valid_assignments),
            observable_names=set(spec.observable_names),
            mechanism_param_names=set(spec.mechanism_param_names),
            scalar_input_names=set(spec.scalar_input_names),
        )
        try:
            scoped_mechanism = _fresh_mechanism(sanitized_source_text, scalar_params, wegscheider=wegscheider)
            _ = apply_parameter_algebra_spec_to_mechanism(
                scoped_spec,
                mechanism=scoped_mechanism,
                require_mutable=False,
            )
        except ValueError as exc:
            return _StepConstraintAuthorityState(
                analysis=_empty_step_constraint_authority_analysis(
                    mechanism_param_names=frozenset(namespace.mechanism_param_names),
                    scalar_input_names=frozenset(namespace.scalar_input_names),
                    observable_names=frozenset(namespace.observable_names),
                    builtin_function_names=frozenset(namespace.builtin_function_names),
                    protected_symbol_names=frozenset(namespace.protected_symbol_names),
                    referenced_scalar_inputs=frozenset(
                        referenced_name
                        for assignment_name, assignment_analysis in analyses_by_name.items()
                        if str(assignment_name) not in affected_assignment_names
                        for referenced_name in getattr(assignment_analysis, "referenced_scalar_inputs", frozenset())
                    ),
                    step_analysis_errors=step_analysis_errors,
                    analysis_error=_build_step_constraint_authority_error(scoped_apply_stage, exc),
                ),
                mechanism=None,
            )

        constrained = (getattr(scoped_mechanism, "metadata", {}) or {}).get("constrained_params") or {}
        reasons: dict[str, str] = {}
        if isinstance(constrained, Mapping):
            for raw_name, raw_info in constrained.items():
                reason = _constraint_reason_from_entry(raw_info)
                if reason:
                    reasons[str(raw_name)] = reason
        for constrained_name in list(reasons):
            step_index = _step_index_from_parameter_name(constrained_name)
            if step_index is not None and int(step_index) in step_analysis_errors:
                reasons.pop(constrained_name, None)
        return _StepConstraintAuthorityState(
            analysis=StepConstraintAuthorityAnalysis(
                constraint_reasons=dict(reasons),
                analysis_error=None,
                step_analysis_errors=dict(step_analysis_errors),
                mechanism_param_names=frozenset(namespace.mechanism_param_names),
                scalar_input_names=frozenset(namespace.scalar_input_names),
                observable_names=frozenset(namespace.observable_names),
                builtin_function_names=frozenset(namespace.builtin_function_names),
                protected_symbol_names=frozenset(namespace.protected_symbol_names),
                referenced_scalar_inputs=frozenset(
                    referenced_name
                    for assignment_name, assignment_analysis in analyses_by_name.items()
                    if str(assignment_name) not in affected_assignment_names
                    for referenced_name in getattr(assignment_analysis, "referenced_scalar_inputs", frozenset())
                ),
            ),
            mechanism=scoped_mechanism,
        )

    symtab = SymbolTable()
    builtin_function_names = frozenset(symtab.functions().keys())
    protected_symbol_names = frozenset(symtab.protected_names())
    mechanism_text = strip_named_reaction_dsl_initial_concentration_sets(str(source_text or ""))
    if not mechanism_text.strip():
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                builtin_function_names=builtin_function_names,
                protected_symbol_names=protected_symbol_names,
            ),
            mechanism=None,
        )

    sanitized_text, scalar_values = _extract_top_level_scalar_assignments(mechanism_text)
    if not sanitized_text.strip():
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                scalar_input_names=frozenset(scalar_values),
                builtin_function_names=builtin_function_names,
                protected_symbol_names=protected_symbol_names,
            ),
            mechanism=None,
        )
    if not _text_has_potential_step_constraints(mechanism_text) and not wegscheider_enabled:
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                scalar_input_names=frozenset(scalar_values),
                builtin_function_names=builtin_function_names,
                protected_symbol_names=protected_symbol_names,
            ),
            mechanism=None,
        )

    try:
        mechanism = _fresh_mechanism(sanitized_text, scalar_values, wegscheider=wegscheider_enabled)
    except ValueError as exc:
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                scalar_input_names=frozenset(scalar_values),
                builtin_function_names=builtin_function_names,
                protected_symbol_names=protected_symbol_names,
                analysis_error=_build_step_constraint_authority_error("parse_mechanism", exc),
            ),
            mechanism=None,
        )

    mechanism_names = set(mechanism_parameter_names(mechanism))
    try:
        spec = parse_parameter_algebra_spec_from_dsl_text(
            mechanism_text,
            mechanism_param_names=mechanism_names,
            scalar_input_names=set(scalar_values),
        )
    except ValueError as exc:
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                mechanism_param_names=frozenset(mechanism_names),
                scalar_input_names=frozenset(scalar_values),
                builtin_function_names=builtin_function_names,
                protected_symbol_names=protected_symbol_names,
                analysis_error=_build_step_constraint_authority_error("parse_parameter_algebra_spec", exc),
            ),
            mechanism=None,
        )

    namespace = spec.namespace_model()
    base_values = read_mechanism_parameter_values(mechanism, names=set(namespace.mechanism_param_names))
    base_values.update({str(name): float(value) for name, value in scalar_values.items()})

    assignment_analyses = analyze_parameter_algebra_assignments(
        spec,
        assignments=spec.param_statements,
        base_values=base_values,
    )
    direct_assignment_errors = {
        str(assignment_name): _build_step_constraint_authority_error(
            "build_parameter_algebra_evaluation_model",
            assignment_analysis.error,
        )
        for assignment_name, assignment_analysis in assignment_analyses.items()
        if getattr(assignment_analysis, "error", None) is not None
    }
    if direct_assignment_errors:
        affected_assignment_names = _collect_affected_assignment_names(
            assignment_analyses,
            set(direct_assignment_errors),
        )
        step_analysis_errors = _build_step_analysis_errors(
            assignment_analyses,
            direct_assignment_errors,
            affected_assignment_names,
        )
        return _state_from_scoped_failures(
            spec=spec,
            namespace=namespace,
            analyses_by_name=assignment_analyses,
            affected_assignment_names=affected_assignment_names,
            step_analysis_errors=step_analysis_errors,
            scoped_apply_stage="apply_parameter_algebra_spec",
            sanitized_source_text=sanitized_text,
            scalar_params=scalar_values,
            wegscheider=wegscheider_enabled,
        )

    try:
        evaluation_model = build_parameter_algebra_evaluation_model(
            spec,
            assignments=spec.param_statements,
            base_values=base_values,
        )
    except ValueError as exc:
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                mechanism_param_names=frozenset(namespace.mechanism_param_names),
                scalar_input_names=frozenset(namespace.scalar_input_names),
                observable_names=frozenset(namespace.observable_names),
                builtin_function_names=frozenset(namespace.builtin_function_names),
                protected_symbol_names=frozenset(namespace.protected_symbol_names),
                analysis_error=_build_step_constraint_authority_error("build_parameter_algebra_evaluation_model", exc),
            ),
            mechanism=None,
        )

    try:
        _ = apply_parameter_algebra_spec_to_mechanism(spec, mechanism=mechanism, require_mutable=False)
    except ValueError as exc:
        failed_assignment_name = _assignment_name_from_dsl_error(tuple(spec.param_statements), exc)
        if failed_assignment_name is not None:
            direct_assignment_errors = {
                str(failed_assignment_name): _build_step_constraint_authority_error("apply_parameter_algebra_spec", exc),
            }
            affected_assignment_names = _collect_affected_assignment_names(
                assignment_analyses,
                set(direct_assignment_errors),
            )
            step_analysis_errors = _build_step_analysis_errors(
                assignment_analyses,
                direct_assignment_errors,
                affected_assignment_names,
            )
            return _state_from_scoped_failures(
                spec=spec,
                namespace=namespace,
                analyses_by_name=assignment_analyses,
                affected_assignment_names=affected_assignment_names,
                step_analysis_errors=step_analysis_errors,
                scoped_apply_stage="apply_parameter_algebra_spec",
                sanitized_source_text=sanitized_text,
                scalar_params=scalar_values,
                wegscheider=wegscheider_enabled,
            )
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                mechanism_param_names=frozenset(namespace.mechanism_param_names),
                scalar_input_names=frozenset(namespace.scalar_input_names),
                observable_names=frozenset(namespace.observable_names),
                builtin_function_names=frozenset(namespace.builtin_function_names),
                protected_symbol_names=frozenset(namespace.protected_symbol_names),
                referenced_scalar_inputs=frozenset(evaluation_model.referenced_scalar_inputs),
                analysis_error=_build_step_constraint_authority_error("apply_parameter_algebra_spec", exc),
            ),
            mechanism=None,
        )

    constrained = (getattr(mechanism, "metadata", {}) or {}).get("constrained_params") or {}
    if not isinstance(constrained, Mapping):
        return _StepConstraintAuthorityState(
            analysis=_empty_step_constraint_authority_analysis(
                mechanism_param_names=frozenset(namespace.mechanism_param_names),
                scalar_input_names=frozenset(namespace.scalar_input_names),
                observable_names=frozenset(namespace.observable_names),
                builtin_function_names=frozenset(namespace.builtin_function_names),
                protected_symbol_names=frozenset(namespace.protected_symbol_names),
                referenced_scalar_inputs=frozenset(evaluation_model.referenced_scalar_inputs),
            ),
            mechanism=mechanism,
        )

    reasons: dict[str, str] = {}
    for raw_name, raw_info in constrained.items():
        reason = _constraint_reason_from_entry(raw_info)
        if reason:
            reasons[str(raw_name)] = reason
    return _StepConstraintAuthorityState(
        analysis=StepConstraintAuthorityAnalysis(
            constraint_reasons=dict(reasons),
            analysis_error=None,
            step_analysis_errors={},
            mechanism_param_names=frozenset(namespace.mechanism_param_names),
            scalar_input_names=frozenset(namespace.scalar_input_names),
            observable_names=frozenset(namespace.observable_names),
            builtin_function_names=frozenset(namespace.builtin_function_names),
            protected_symbol_names=frozenset(namespace.protected_symbol_names),
            referenced_scalar_inputs=frozenset(evaluation_model.referenced_scalar_inputs),
        ),
        mechanism=mechanism,
    )


def build_step_constraint_authority_analysis_from_text(
    source_text: str,
    *,
    context: Mapping[str, object] | None = None,
) -> StepConstraintAuthorityAnalysis:
    return _build_step_constraint_authority_state_from_text(
        source_text,
        context=context,
    ).analysis


def read_step_equilibrium_authoritative_values_from_text(
    source_text: str,
    *,
    step_index: int,
    context: Mapping[str, object] | None = None,
) -> dict[str, float]:
    from kindred.core.simulator.parameter_algebra import read_mechanism_parameter_values

    state = _build_step_constraint_authority_state_from_text(
        source_text,
        context=context,
    )
    analysis = state.analysis
    if analysis.analysis_error is not None or analysis.step_analysis_errors.get(int(step_index)) is not None:
        raise StepConstraintAuthorityUnavailable(
            step_index=int(step_index),
            analysis=analysis,
        )
    if state.mechanism is None:
        raise StepConstraintAuthorityUnavailable(
            step_index=int(step_index),
            analysis=analysis,
        )
    return read_mechanism_parameter_values(
        state.mechanism,
        names={f"kf{int(step_index)}", f"kr{int(step_index)}", f"K{int(step_index)}"},
    )


def build_step_constraint_reasons_from_text(
    source_text: str,
    *,
    context: Mapping[str, object] | None = None,
) -> dict[str, str]:
    return dict(
        build_step_constraint_authority_analysis_from_text(
            source_text,
            context=context,
        ).constraint_reasons
    )
