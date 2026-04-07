"""
Small, pure helpers for editing Kindred DSL text.

This module intentionally avoids importing the DSL parser to keep it cycle-safe and
cheap to import from GUI/controller layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Mapping, Tuple

from .step_constraint_authority import (
    StepConstraintAuthorityAnalysis,
    StepConstraintAuthorityError,
    StepConstraintAuthorityUnavailable,
    build_step_constraint_authority_analysis_from_text,
    read_step_equilibrium_authoritative_values_from_text,
)


AUTHORITATIVE_PARAMETER_SIG_DIGITS = 15
_STEP_PARAMETER_RE = re.compile(r"^(kf|kr|K|k)\d+$")
_STEP_PARAMETER_FLOOR = 1e-12
_EQUILIBRIUM_K_ALIAS_LOWER = frozenset({"keq", "k_eq"})


__all__ = [
    "AUTHORITATIVE_PARAMETER_SIG_DIGITS",
    "CurrentTextStepAnalysisContext",
    "ParameterTextUpdateAnalysis",
    "StepParameterUpdateOutcome",
    "analyze_parameter_updates_to_dsl_text",
    "analyze_step_parameter_update",
    "apply_parameter_updates_to_dsl_text",
    "authoritative_parameter_change_name_aware",
    "build_current_text_step_analysis_context",
    "authoritative_parameter_values_match",
    "format_authoritative_parameter_value",
    "step_rewrite_block_reason",
]


@dataclass(frozen=True)
class StepParameterUpdateOutcome:
    parameter_name: str
    parameter_family: str
    step_index: int
    found_target: bool
    writable: bool
    requested_value: float
    effective_authoritative_written_value: float | None
    semantic_value_change: bool
    would_change_text: bool
    canonicalization_only_change: bool
    updated_text: str
    warning_reason: str | None
    line_index: int | None
    line_prefix: str | None
    resolved_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ParameterTextUpdateAnalysis:
    updated_text: str
    missing: tuple[str, ...]
    update_errors: tuple[dict[str, str], ...]
    step_outcomes: tuple[StepParameterUpdateOutcome, ...]


@dataclass(frozen=True)
class CurrentTextStepAnalysisContext:
    source_text: str
    temperature_K: float
    wegscheider_cyclicity_enabled: bool
    lines: tuple[str, ...]
    reaction_lines: Mapping[int, int]
    equilibrium_lines: Mapping[int, int]
    step_constraint_analysis: StepConstraintAuthorityAnalysis

    @property
    def step_constraint_reasons(self) -> Mapping[str, str]:
        return self.step_constraint_analysis.constraint_reasons

    @property
    def constraint_analysis_error(self) -> StepConstraintAuthorityError | None:
        return self.step_constraint_analysis.analysis_error

    @property
    def step_constraint_analysis_errors(self) -> Mapping[int, StepConstraintAuthorityError]:
        return self.step_constraint_analysis.step_analysis_errors


def _normalized_step_constraint_context_values(context: Mapping[str, object] | None) -> tuple[float, bool]:
    temperature_K = 298.15
    if isinstance(context, Mapping):
        try:
            temperature_K = float(context.get("temperature_K", temperature_K))
        except (TypeError, ValueError):
            temperature_K = 298.15
        wegscheider_enabled = bool(context.get("wegscheider_cyclicity_enabled", False))
    else:
        wegscheider_enabled = False
    return float(temperature_K), bool(wegscheider_enabled)


def _normalize_authoritative_parameter_float(value: object) -> float:
    normalized = float(value)
    if normalized == 0.0:
        return 0.0
    return normalized


def format_authoritative_parameter_value(value: object) -> str:
    normalized = _normalize_authoritative_parameter_float(value)
    return f"{normalized:.{AUTHORITATIVE_PARAMETER_SIG_DIGITS}g}"


def authoritative_parameter_values_match(current_value: object, target_value: object) -> bool:
    try:
        return format_authoritative_parameter_value(current_value) == format_authoritative_parameter_value(target_value)
    except (TypeError, ValueError):
        return False


def authoritative_parameter_change_name_aware(
    name: object,
    current_value: object,
    target_value: object,
    *,
    source_text: str | None = None,
    canonical_updater: Callable[[str, float, str], str] | None = None,
) -> bool:
    name_str = str(name)
    if source_text is not None and _STEP_PARAMETER_RE.match(name_str):
        if canonical_updater is None:
            try:
                outcome = analyze_step_parameter_update(
                    str(source_text or ""),
                    name_str,
                    target_value,
                    authoritative_current_value=current_value,
                )
            except (TypeError, ValueError):
                return True
            return bool(outcome.semantic_value_change or outcome.canonicalization_only_change)
    if (
        canonical_updater is not None
        and source_text is not None
        and _STEP_PARAMETER_RE.match(name_str)
    ):
        try:
            updated_text, _, update_errors = apply_parameter_updates_to_dsl_text(
                str(source_text or ""),
                {name_str: target_value},
                canonical_updater=canonical_updater,
            )
        except (TypeError, ValueError):
            return True
        if update_errors:
            return True
        return str(updated_text) != str(source_text or "")
    return not authoritative_parameter_values_match(current_value, target_value)


def _parse_mechanism_semicolon_kv(line: str) -> tuple[str, list[list[str]], str]:
    before_comment, sep, comment = str(line or "").partition("#")
    comment_tail = f"{sep}{comment}" if sep else ""
    prefix, sep_params, rest = before_comment.partition(";")
    prefix = prefix.rstrip()
    tokens: list[list[str]] = []
    if sep_params:
        for token in re.split(r"[;,]", rest):
            token = token.strip()
            if not token:
                continue
            key, _, val = token.partition("=")
            tokens.append([key.strip(), val.strip()])
    return prefix, tokens, comment_tail


def _serialize_mechanism_semicolon_kv(prefix: str, tokens: list[list[str]], comment_tail: str) -> str:
    if tokens:
        params = ", ".join(f"{key}={val}" if val else f"{key}=" for key, val in tokens)
        base = f"{prefix} ; {params}"
    else:
        base = prefix
    if comment_tail:
        base = f"{base} {comment_tail.strip()}"
    return base.strip()


def _dedupe_tokens_case_insensitive(tokens: list[list[str]]) -> list[list[str]]:
    seen: set[str] = set()
    result: list[list[str]] = []
    for key, val in tokens:
        lower = str(key).lower()
        if lower in seen:
            continue
        seen.add(lower)
        result.append([key, val])
    return result


def _is_equilibrium_k_token(key: object) -> bool:
    key_str = str(key).strip()
    return key_str == "K" or key_str.lower() in _EQUILIBRIUM_K_ALIAS_LOWER


def _token_matches_alias(key: object, alias: str) -> bool:
    if str(alias) == "K":
        return _is_equilibrium_k_token(key)
    alias_str = str(alias)
    if _is_equilibrium_k_token(key):
        return False
    return str(key).strip().lower() == alias_str.lower()


def _canonical_step_token_key_for_duplicate_check(key: object) -> str | None:
    key_str = str(key).strip()
    if not key_str:
        return None
    if _is_equilibrium_k_token(key_str):
        return "K"
    lowered = key_str.lower()
    if lowered in {"k", "kf", "kr"}:
        return lowered
    return None


def _duplicate_canonical_step_token(tokens: list[list[str]]) -> tuple[str, str, str] | None:
    seen: dict[str, str] = {}
    for key, _ in tokens:
        raw_key = str(key).strip()
        canonical_key = _canonical_step_token_key_for_duplicate_check(raw_key)
        if canonical_key is None:
            continue
        previous_spelling = seen.get(canonical_key)
        if previous_spelling is not None:
            return previous_spelling, raw_key, canonical_key
        seen[canonical_key] = raw_key
    return None


def _raise_on_duplicate_canonical_step_tokens(tokens: list[list[str]]) -> None:
    duplicate = _duplicate_canonical_step_token(tokens)
    if duplicate is None:
        return
    previous_spelling, raw_key, canonical_key = duplicate
    raise ValueError(
        f"Duplicate parameter: '{previous_spelling}' and '{raw_key}' both resolve to {canonical_key}"
    )


def _get_token_float(tokens: list[list[str]], aliases: tuple[str, ...], default: float | None = None) -> float | None:
    for key, val in tokens:
        if any(_token_matches_alias(key, alias) for alias in aliases):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
    return default


def _has_token_alias(tokens: list[list[str]], aliases: tuple[str, ...]) -> bool:
    for key, _ in tokens:
        if any(_token_matches_alias(key, alias) for alias in aliases):
            return True
    return False


def _set_token_float(
    tokens: list[list[str]],
    canonical_key: str,
    float_value: float,
    *,
    aliases: tuple[str, ...] = (),
    sig: int | None = None,
) -> None:
    if sig is None:
        sanitized = format_authoritative_parameter_value(float_value)
    else:
        sanitized = f"{float(float_value):.{int(sig)}g}"

    if canonical_key == "K":
        target_index = None
        for idx, (key, _) in enumerate(tokens):
            if _is_equilibrium_k_token(key):
                target_index = idx
                break
        if target_index is not None:
            tokens[target_index][0] = canonical_key
            tokens[target_index][1] = sanitized
        else:
            tokens.append([canonical_key, sanitized])
            target_index = len(tokens) - 1
        for idx in range(len(tokens) - 1, -1, -1):
            if idx == target_index:
                continue
            if _is_equilibrium_k_token(tokens[idx][0]):
                tokens.pop(idx)
        return

    exact_index = None
    for idx, (key, _) in enumerate(tokens):
        if _token_matches_alias(key, canonical_key):
            exact_index = idx
            break

    if exact_index is not None:
        target_index = exact_index
        tokens[target_index][0] = canonical_key
        tokens[target_index][1] = sanitized
    else:
        tokens.append([canonical_key, sanitized])
        target_index = len(tokens) - 1

    for idx in range(len(tokens) - 1, -1, -1):
        if idx == target_index:
            continue
        if any(_token_matches_alias(tokens[idx][0], alias) for alias in (canonical_key, *aliases)):
            tokens.pop(idx)


def _remove_token_aliases(tokens: list[list[str]], aliases: tuple[str, ...]) -> None:
    filtered: list[list[str]] = []
    for key, val in tokens:
        if not any(_token_matches_alias(key, alias) for alias in aliases):
            filtered.append([key, val])
    tokens[:] = filtered


def _index_step_lines(lines: list[str]) -> tuple[dict[int, int], dict[int, int]]:
    reaction_lines: dict[int, int] = {}
    equilibrium_lines: dict[int, int] = {}
    line_counter = 0
    for idx, line in enumerate(lines):
        stripped = str(line).strip()
        if not stripped or stripped.startswith("#"):
            continue
        lower = stripped.lower()
        if "<->" in lower or "<=>" in lower:
            line_counter += 1
            equilibrium_lines[line_counter] = idx
            continue
        if "->" in lower:
            line_counter += 1
            reaction_lines[line_counter] = idx
    return reaction_lines, equilibrium_lines


def build_current_text_step_analysis_context(
    source_text: str,
    *,
    step_constraint_context: Mapping[str, object] | None = None,
) -> CurrentTextStepAnalysisContext:
    normalized_text = str(source_text or "")
    lines = normalized_text.split("\n")
    reaction_lines, equilibrium_lines = _index_step_lines(lines)
    temperature_K, wegscheider_enabled = _normalized_step_constraint_context_values(step_constraint_context)
    step_constraint_analysis = build_step_constraint_authority_analysis_from_text(
        normalized_text,
        context={
            "temperature_K": temperature_K,
            "wegscheider_cyclicity_enabled": wegscheider_enabled,
        },
    )
    return CurrentTextStepAnalysisContext(
        source_text=normalized_text,
        temperature_K=temperature_K,
        wegscheider_cyclicity_enabled=wegscheider_enabled,
        lines=tuple(lines),
        reaction_lines=dict(reaction_lines),
        equilibrium_lines=dict(equilibrium_lines),
        step_constraint_analysis=step_constraint_analysis,
    )


def _coerce_current_text_step_analysis_context(
    source_text: str,
    *,
    step_analysis_context: CurrentTextStepAnalysisContext | None = None,
    step_constraint_context: Mapping[str, object] | None = None,
) -> CurrentTextStepAnalysisContext:
    normalized_text = str(source_text or "")
    temperature_K, wegscheider_enabled = _normalized_step_constraint_context_values(step_constraint_context)
    if (
        isinstance(step_analysis_context, CurrentTextStepAnalysisContext)
        and step_analysis_context.source_text == normalized_text
        and step_analysis_context.temperature_K == temperature_K
        and step_analysis_context.wegscheider_cyclicity_enabled == wegscheider_enabled
    ):
        return step_analysis_context
    return build_current_text_step_analysis_context(
        normalized_text,
        step_constraint_context={
            "temperature_K": temperature_K,
            "wegscheider_cyclicity_enabled": wegscheider_enabled,
        },
    )


def _parse_step_parameter_name(name: str) -> tuple[str, int] | None:
    match = _STEP_PARAMETER_RE.match(str(name))
    if not match:
        return None
    family = str(match.group(1))
    suffix = str(name)[len(family):]
    try:
        return family, int(suffix)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _coerce_required_finite_float(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Step parameter value must be finite.")
    return converted


def _normalize_k_value(value: float) -> float:
    normalized = float(value)
    if normalized == 0.0:
        normalized = _STEP_PARAMETER_FLOOR
    if normalized < 0.0:
        normalized = -normalized
    return normalized


def _normalize_rate_value(value: float) -> float:
    normalized = abs(float(value))
    if normalized < _STEP_PARAMETER_FLOOR:
        normalized = _STEP_PARAMETER_FLOOR
    return normalized


def _derive_equilibrium_role_from_tokens(tokens: list[list[str]]) -> str:
    if not _has_token_alias(tokens, ("K",)):
        return ""
    user_kf_explicit = _has_token_alias(tokens, ("kf", "k"))
    user_kr_explicit = _has_token_alias(tokens, ("kr",))
    if user_kr_explicit and not user_kf_explicit:
        return "kf"
    return "kr"


def _derive_equilibrium_role(
    tokens: list[list[str]],
    *,
    step_index: int,
    step_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    _ = step_index
    _ = step_metadata
    return _derive_equilibrium_role_from_tokens(tokens)


def _current_effective_step_value(
    family: str,
    tokens: list[list[str]],
    *,
    has_explicit_k: bool,
) -> float | None:
    if family == "k":
        return _coerce_optional_float(_get_token_float(tokens, ("k",), None))
    if family == "K":
        explicit_k = _coerce_optional_float(_get_token_float(tokens, ("K",), None))
        if explicit_k is not None:
            return _normalize_k_value(explicit_k)
        kf_val = _coerce_optional_float(_get_token_float(tokens, ("kf", "k"), None))
        kr_val = _coerce_optional_float(_get_token_float(tokens, ("kr",), None))
        if kf_val is None or kr_val is None or abs(kr_val) < _STEP_PARAMETER_FLOOR:
            return None
        return _normalize_k_value(kf_val / kr_val)
    if family == "kf":
        kf_val = _coerce_optional_float(_get_token_float(tokens, ("kf", "k"), None))
        if kf_val is not None:
            return _normalize_rate_value(kf_val)
        if has_explicit_k:
            kr_val = _coerce_optional_float(_get_token_float(tokens, ("kr",), None))
            k_val = _coerce_optional_float(_get_token_float(tokens, ("K",), None))
            if kr_val is None or k_val is None:
                return None
            return _normalize_rate_value(kr_val * _normalize_k_value(k_val))
        return None
    if family == "kr":
        kr_val = _coerce_optional_float(_get_token_float(tokens, ("kr",), None))
        if kr_val is not None:
            return _normalize_rate_value(kr_val)
        if has_explicit_k:
            kf_val = _coerce_optional_float(_get_token_float(tokens, ("kf", "k"), None))
            k_val = _coerce_optional_float(_get_token_float(tokens, ("K",), None))
            if kf_val is None or k_val is None:
                return None
            normalized_k = _normalize_k_value(k_val)
            if abs(normalized_k) < _STEP_PARAMETER_FLOOR:
                return None
            return _normalize_rate_value(kf_val / normalized_k)
    return None


def step_rewrite_block_reason(
    *,
    step_index: int,
    affected_parameter_names: tuple[str, ...],
    step_analysis_context: CurrentTextStepAnalysisContext,
) -> str | None:
    if step_analysis_context.step_constraint_analysis_errors.get(step_index) is not None:
        return "constraint_analysis_failed"
    if step_analysis_context.constraint_analysis_error is not None:
        return "constraint_analysis_failed"
    current_step_constraints = step_analysis_context.step_constraint_reasons
    for affected_name in affected_parameter_names:
        if current_step_constraints.get(str(affected_name)):
            return "target_unwritable"
    return None


def _semantic_equilibrium_changed_constrained_targets(
    *,
    source_text: str,
    updated_text: str,
    step_index: int,
    step_analysis_context: CurrentTextStepAnalysisContext,
) -> tuple[str, ...]:
    constrained_family_names = tuple(
        name
        for name in (f"kf{step_index}", f"kr{step_index}", f"K{step_index}")
        if step_analysis_context.step_constraint_reasons.get(name)
    )
    if not constrained_family_names:
        return ()
    authority_context = {
        "temperature_K": float(step_analysis_context.temperature_K),
        "wegscheider_cyclicity_enabled": bool(step_analysis_context.wegscheider_cyclicity_enabled),
    }
    before_values = read_step_equilibrium_authoritative_values_from_text(
        source_text,
        step_index=step_index,
        context=authority_context,
    )
    after_values = read_step_equilibrium_authoritative_values_from_text(
        updated_text,
        step_index=step_index,
        context=authority_context,
    )
    changed_names: list[str] = []
    for constrained_name in constrained_family_names:
        before_value = before_values.get(constrained_name)
        after_value = after_values.get(constrained_name)
        if before_value is None and after_value is None:
            continue
        if before_value is None or after_value is None:
            changed_names.append(constrained_name)
            continue
        if not authoritative_parameter_values_match(before_value, after_value):
            changed_names.append(constrained_name)
    return tuple(changed_names)


def analyze_step_parameter_update(
    source_text: str,
    name: str,
    requested_value: object,
    *,
    authoritative_current_value: object | None = None,
    step_metadata: Mapping[str, Mapping[str, object]] | None = None,
    step_constraint_context: Mapping[str, object] | None = None,
    step_analysis_context: CurrentTextStepAnalysisContext | None = None,
) -> StepParameterUpdateOutcome:
    parsed = _parse_step_parameter_name(str(name))
    if parsed is None:
        raise ValueError(f"Unsupported step parameter name: {name!r}")

    family, step_index = parsed
    requested_float = _coerce_required_finite_float(requested_value)
    original_text = str(source_text or "")
    current_text_context = _coerce_current_text_step_analysis_context(
        original_text,
        step_analysis_context=step_analysis_context,
        step_constraint_context=step_constraint_context,
    )
    lines = list(current_text_context.lines)
    reaction_lines = current_text_context.reaction_lines
    equilibrium_lines = current_text_context.equilibrium_lines

    if family == "k":
        line_index = reaction_lines.get(step_index)
        if line_index is None:
            return StepParameterUpdateOutcome(
                parameter_name=str(name),
                parameter_family=family,
                step_index=step_index,
                found_target=False,
                writable=False,
                requested_value=requested_float,
                effective_authoritative_written_value=None,
                semantic_value_change=False,
                would_change_text=False,
                canonicalization_only_change=False,
                updated_text=original_text,
                warning_reason="missing_target",
                line_index=None,
                line_prefix=None,
                resolved_values=(),
            )
        prefix, tokens, comment = _parse_mechanism_semicolon_kv(lines[line_index])
        _raise_on_duplicate_canonical_step_tokens(tokens)
        tokens = _dedupe_tokens_case_insensitive(tokens)
        current_step_constraints = current_text_context.step_constraint_reasons
        current_effective = _current_effective_step_value(family, tokens, has_explicit_k=False)
        baseline_value = _coerce_optional_float(authoritative_current_value)
        if baseline_value is None:
            baseline_value = current_effective
        warning_reason = None
        if current_text_context.step_constraint_analysis_errors.get(step_index) is not None:
            warning_reason = "constraint_analysis_failed"
        elif current_text_context.constraint_analysis_error is not None:
            warning_reason = "constraint_analysis_failed"
        elif current_step_constraints.get(f"k{step_index}"):
            warning_reason = "target_unwritable"
        if warning_reason is not None:
            return StepParameterUpdateOutcome(
                parameter_name=str(name),
                parameter_family=family,
                step_index=step_index,
                found_target=True,
                writable=False,
                requested_value=requested_float,
                effective_authoritative_written_value=baseline_value,
                semantic_value_change=False,
                would_change_text=False,
                canonicalization_only_change=False,
                updated_text=original_text,
                warning_reason=warning_reason,
                line_index=int(line_index),
                line_prefix=prefix.strip(),
                resolved_values=(),
            )
        _set_token_float(tokens, "k", requested_float)
        _remove_token_aliases(tokens, ("kf", "kr"))
        new_lines = list(lines)
        new_lines[line_index] = _serialize_mechanism_semicolon_kv(prefix, tokens, comment)
        updated_text = "\n".join(new_lines)
        effective_value = requested_float
        would_change_text = updated_text != original_text
        semantic_value_change = (
            would_change_text if baseline_value is None else not authoritative_parameter_values_match(baseline_value, effective_value)
        )
        return StepParameterUpdateOutcome(
            parameter_name=str(name),
            parameter_family=family,
            step_index=step_index,
            found_target=True,
            writable=True,
            requested_value=requested_float,
            effective_authoritative_written_value=effective_value,
            semantic_value_change=bool(semantic_value_change),
            would_change_text=bool(would_change_text),
            canonicalization_only_change=bool(would_change_text and not semantic_value_change),
            updated_text=updated_text,
            warning_reason=None,
            line_index=int(line_index),
            line_prefix=prefix.strip(),
            resolved_values=((f"k{step_index}", effective_value),),
        )

    line_index = equilibrium_lines.get(step_index)
    if line_index is None:
        return StepParameterUpdateOutcome(
            parameter_name=str(name),
            parameter_family=family,
            step_index=step_index,
            found_target=False,
            writable=False,
            requested_value=requested_float,
            effective_authoritative_written_value=None,
            semantic_value_change=False,
            would_change_text=False,
            canonicalization_only_change=False,
            updated_text=original_text,
            warning_reason="missing_target",
            line_index=None,
            line_prefix=None,
            resolved_values=(),
        )

    prefix, tokens, comment = _parse_mechanism_semicolon_kv(lines[line_index])
    _raise_on_duplicate_canonical_step_tokens(tokens)
    tokens = _dedupe_tokens_case_insensitive(tokens)
    has_explicit_k = _has_token_alias(tokens, ("K",))
    derive_rate = _derive_equilibrium_role(tokens, step_index=step_index, step_metadata=step_metadata)
    current_step_constraints = current_text_context.step_constraint_reasons
    current_effective = _current_effective_step_value(family, tokens, has_explicit_k=has_explicit_k)
    baseline_value = _coerce_optional_float(authoritative_current_value)
    if baseline_value is None:
        baseline_value = current_effective

    writable = True
    warning_reason = None
    non_k_block_reason = current_step_constraints.get(f"{family}{step_index}")
    if current_text_context.step_constraint_analysis_errors.get(step_index) is not None:
        writable = False
        warning_reason = "constraint_analysis_failed"
    elif current_text_context.constraint_analysis_error is not None:
        writable = False
        warning_reason = "constraint_analysis_failed"
    elif non_k_block_reason:
        writable = False
        warning_reason = "target_unwritable"
    elif family == "K" and not has_explicit_k:
        writable = False
        warning_reason = "target_unwritable"
    elif family == "kf" and has_explicit_k and derive_rate == "kf":
        writable = False
        warning_reason = "target_unwritable"
    elif family == "kr" and has_explicit_k and derive_rate == "kr":
        writable = False
        warning_reason = "target_unwritable"

    if not writable:
        return StepParameterUpdateOutcome(
            parameter_name=str(name),
            parameter_family=family,
            step_index=step_index,
            found_target=True,
            writable=False,
            requested_value=requested_float,
            effective_authoritative_written_value=baseline_value,
            semantic_value_change=False,
            would_change_text=False,
            canonicalization_only_change=False,
            updated_text=original_text,
            warning_reason=warning_reason,
            line_index=int(line_index),
            line_prefix=prefix.strip(),
            resolved_values=(),
        )

    working_tokens = [list(token) for token in tokens]
    resolved_values: list[tuple[str, float]] = []

    if family == "K":
        k_value = _normalize_k_value(requested_float)
        if derive_rate == "kf":
            kr_value = _current_effective_step_value("kr", working_tokens, has_explicit_k=has_explicit_k)
            if kr_value is None:
                kr_value = 1.0
            kr_value = _normalize_rate_value(kr_value)
            kf_value = _normalize_rate_value(kr_value * k_value)
            _set_token_float(working_tokens, "K", k_value)
            _set_token_float(working_tokens, "kr", kr_value, aliases=("kr",))
            _remove_token_aliases(working_tokens, ("kf", "k"))
        else:
            kf_value = _current_effective_step_value("kf", working_tokens, has_explicit_k=has_explicit_k)
            if kf_value is None:
                kf_value = 1.0
            kf_value = _normalize_rate_value(kf_value)
            kr_value = _normalize_rate_value(kf_value / k_value)
            _set_token_float(working_tokens, "K", k_value)
            _set_token_float(working_tokens, "kf", kf_value, aliases=("k",))
            _remove_token_aliases(working_tokens, ("kr",))
        effective_value = k_value
        resolved_values.extend(
            (
                (f"K{step_index}", k_value),
                (f"kf{step_index}", kf_value),
                (f"kr{step_index}", kr_value),
            )
        )
    elif family == "kf":
        kf_value = _normalize_rate_value(requested_float)
        if not has_explicit_k:
            _remove_token_aliases(working_tokens, ("K",))
        _set_token_float(working_tokens, "kf", kf_value, aliases=("k",))

        kr_value = _coerce_optional_float(_get_token_float(working_tokens, ("kr",), None))
        k_value = _coerce_optional_float(_get_token_float(working_tokens, ("K",), None))
        k_valid = k_value is not None and abs(k_value) > _STEP_PARAMETER_FLOOR
        kr_valid = kr_value is not None and abs(kr_value) > _STEP_PARAMETER_FLOOR

        if has_explicit_k and k_valid:
            normalized_k = _normalize_k_value(float(k_value))
            kr_value = _normalize_rate_value(kf_value / normalized_k)
            _set_token_float(working_tokens, "K", normalized_k)
            if derive_rate == "kr":
                _remove_token_aliases(working_tokens, ("kr",))
            else:
                _set_token_float(working_tokens, "kr", kr_value, aliases=("kr",))
            k_value = normalized_k
        else:
            if not kr_valid:
                kr_value = max(kf_value, _STEP_PARAMETER_FLOOR)
                _set_token_float(working_tokens, "kr", kr_value, aliases=("kr",))
            else:
                kr_value = _normalize_rate_value(float(kr_value))
            if has_explicit_k:
                k_value = _normalize_k_value(kf_value / kr_value)
                _set_token_float(working_tokens, "K", k_value)
        effective_value = kf_value
        resolved_values.append((f"kf{step_index}", kf_value))
        if kr_value is not None:
            resolved_values.append((f"kr{step_index}", kr_value))
        if has_explicit_k and k_value is not None:
            resolved_values.append((f"K{step_index}", k_value))
    else:
        kr_value = _normalize_rate_value(requested_float)
        if not has_explicit_k:
            _remove_token_aliases(working_tokens, ("K",))
        _set_token_float(working_tokens, "kr", kr_value, aliases=("kr",))

        k_value = _coerce_optional_float(_get_token_float(working_tokens, ("K",), None)) if has_explicit_k else None
        k_valid = k_value is not None and abs(k_value) > _STEP_PARAMETER_FLOOR

        if has_explicit_k and derive_rate == "kf" and k_valid:
            normalized_k = _normalize_k_value(float(k_value))
            kf_value = _normalize_rate_value(kr_value * normalized_k)
            _set_token_float(working_tokens, "K", normalized_k)
            _remove_token_aliases(working_tokens, ("kf", "k"))
            k_value = normalized_k
        else:
            kf_value = _coerce_optional_float(_get_token_float(working_tokens, ("kf", "k"), None))
            if kf_value is None:
                kf_value = max(kr_value, _STEP_PARAMETER_FLOOR)
            kf_value = _normalize_rate_value(kf_value)
            _set_token_float(working_tokens, "kf", kf_value, aliases=("k",))

            if has_explicit_k:
                k_value = _normalize_k_value(kf_value / kr_value)
                _set_token_float(working_tokens, "K", k_value)
                if derive_rate == "kf":
                    _remove_token_aliases(working_tokens, ("kf", "k"))
        effective_value = kr_value
        resolved_values.append((f"kr{step_index}", kr_value))
        if kf_value is not None:
            resolved_values.append((f"kf{step_index}", kf_value))
        if has_explicit_k and k_value is not None:
            resolved_values.append((f"K{step_index}", k_value))

    new_lines = list(lines)
    new_lines[line_index] = _serialize_mechanism_semicolon_kv(prefix, working_tokens, comment)
    candidate_updated_text = "\n".join(new_lines)
    try:
        semantically_changed_targets = _semantic_equilibrium_changed_constrained_targets(
            source_text=original_text,
            updated_text=candidate_updated_text,
            step_index=step_index,
            step_analysis_context=current_text_context,
        )
    except StepConstraintAuthorityUnavailable:
        affected_warning_reason = "constraint_analysis_failed"
    else:
        affected_warning_reason = step_rewrite_block_reason(
            step_index=step_index,
            affected_parameter_names=semantically_changed_targets,
            step_analysis_context=current_text_context,
        )
    if affected_warning_reason is not None:
        return StepParameterUpdateOutcome(
            parameter_name=str(name),
            parameter_family=family,
            step_index=step_index,
            found_target=True,
            writable=False,
            requested_value=requested_float,
            effective_authoritative_written_value=baseline_value,
            semantic_value_change=False,
            would_change_text=False,
            canonicalization_only_change=False,
            updated_text=original_text,
            warning_reason=affected_warning_reason,
            line_index=int(line_index),
            line_prefix=prefix.strip(),
            resolved_values=(),
        )
    updated_text = candidate_updated_text
    would_change_text = updated_text != original_text
    semantic_value_change = (
        would_change_text if baseline_value is None else not authoritative_parameter_values_match(baseline_value, effective_value)
    )
    return StepParameterUpdateOutcome(
        parameter_name=str(name),
        parameter_family=family,
        step_index=step_index,
        found_target=True,
        writable=True,
        requested_value=requested_float,
        effective_authoritative_written_value=effective_value,
        semantic_value_change=bool(semantic_value_change),
        would_change_text=bool(would_change_text),
        canonicalization_only_change=bool(would_change_text and not semantic_value_change),
        updated_text=updated_text,
        warning_reason=None,
        line_index=int(line_index),
        line_prefix=prefix.strip(),
        resolved_values=tuple(resolved_values),
    )


def analyze_parameter_updates_to_dsl_text(
    source_text: str,
    parameters: Mapping[str, object],
    *,
    authoritative_values: Mapping[str, object] | None = None,
    step_metadata: Mapping[str, Mapping[str, object]] | None = None,
    step_constraint_context: Mapping[str, object] | None = None,
) -> ParameterTextUpdateAnalysis:
    updated_text = str(source_text or "")
    missing: list[str] = []
    update_errors: list[dict[str, str]] = []
    step_outcomes: list[StepParameterUpdateOutcome] = []
    authoritative_map = {str(k): v for k, v in dict(authoritative_values or {}).items()}
    step_analysis_context: CurrentTextStepAnalysisContext | None = None

    for raw_name, raw_value in (parameters or {}).items():
        name = str(raw_name)
        try:
            value = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            message = str(exc).strip() or "Fitted value is invalid."
            update_errors.append(
                {
                    "name": name,
                    "exc_type": exc.__class__.__name__,
                    "message": message,
                }
            )
            continue
        if not math.isfinite(value):
            update_errors.append(
                {
                    "name": name,
                    "exc_type": "ValueError",
                    "message": "Fitted value is non-finite.",
                }
            )
            continue

        if _STEP_PARAMETER_RE.match(name):
            try:
                parsed_name = _parse_step_parameter_name(name)
                needs_step_analysis_context = bool(parsed_name is not None)
                if needs_step_analysis_context and (
                    step_analysis_context is None or step_analysis_context.source_text != updated_text
                ):
                    step_analysis_context = build_current_text_step_analysis_context(
                        updated_text,
                        step_constraint_context=step_constraint_context,
                    )
                outcome = analyze_step_parameter_update(
                    updated_text,
                    name,
                    value,
                    authoritative_current_value=authoritative_map.get(name),
                    step_metadata=step_metadata,
                    step_constraint_context=step_constraint_context,
                    step_analysis_context=step_analysis_context if needs_step_analysis_context else None,
                )
            except Exception as exc:
                update_errors.append(
                    {
                        "name": name,
                        "exc_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
                continue
            step_outcomes.append(outcome)
            if not outcome.found_target:
                missing.append(name)
            updated_text = outcome.updated_text
            if step_analysis_context is not None and step_analysis_context.source_text != updated_text:
                step_analysis_context = None
            continue

        escaped_name = re.escape(name)
        pattern_toplevel = rf"^\s*{escaped_name}\s*=\s*(?P<value>[^\n#;]+)"
        pattern_inline = rf"(?<=[;,])\s*{escaped_name}\s*=\s*(?P<value>[^,\n#]+)"

        matches_toplevel = list(re.finditer(pattern_toplevel, updated_text, re.MULTILINE))
        matches_inline = list(re.finditer(pattern_inline, updated_text))
        if not (matches_toplevel or matches_inline):
            missing.append(name)
            continue

        def _replace_value(match: re.Match[str]) -> str:
            old_value = match.group("value").strip()
            new_value = format_authoritative_parameter_value(value)
            return match.group(0).replace(old_value, new_value, 1)

        if matches_toplevel:
            updated_text = re.sub(pattern_toplevel, _replace_value, updated_text, flags=re.MULTILINE)
        if matches_inline:
            updated_text = re.sub(pattern_inline, _replace_value, updated_text)

    return ParameterTextUpdateAnalysis(
        updated_text=updated_text,
        missing=tuple(missing),
        update_errors=tuple(dict(error) for error in update_errors),
        step_outcomes=tuple(step_outcomes),
    )


def apply_parameter_updates_to_dsl_text(
    source_text: str,
    parameters: Mapping[str, object],
    *,
    canonical_updater: Callable[[str, float, str], str] | None = None,
) -> Tuple[str, list[str], list[dict[str, str]]]:
    """
    Apply parameter value updates to a DSL text blob.

    Parameters
    ----------
    source_text:
        DSL text to update.
    parameters:
        Mapping of {name: value}. Values are coerced to float.
    canonical_updater:
        Optional callable for canonical step-indexed parameters (k1, kf1, kr1, K1).
        Signature: (name, value, text) -> updated_text.

    Returns
    -------
    (updated_text, missing_names, update_errors)
        missing_names includes entries that were not found in the text or had invalid values.
        update_errors includes structured canonical-updater failures that were not
        simple lookup misses.
    """

    if canonical_updater is None:
        analysis = analyze_parameter_updates_to_dsl_text(source_text, parameters)
        return (
            str(analysis.updated_text),
            [str(name) for name in analysis.missing],
            [dict(error) for error in analysis.update_errors],
        )

    updated_text = str(source_text or "")
    missing: list[str] = []
    update_errors: list[dict[str, str]] = []

    for raw_name, raw_value in (parameters or {}).items():
        name = str(raw_name)
        try:
            value = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            missing.append(name)
            continue

        if canonical_updater is not None and _STEP_PARAMETER_RE.match(name):
            try:
                updated_text = str(canonical_updater(name, float(value), updated_text))
            except LookupError:
                missing.append(name)
            except Exception as exc:
                update_errors.append(
                    {
                        "name": name,
                        "exc_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
            continue

        escaped_name = re.escape(name)
        pattern_toplevel = rf"^\s*{escaped_name}\s*=\s*(?P<value>[^\n#;]+)"
        pattern_inline = rf"(?<=[;,])\s*{escaped_name}\s*=\s*(?P<value>[^,\n#]+)"

        matches_toplevel = list(re.finditer(pattern_toplevel, updated_text, re.MULTILINE))
        matches_inline = list(re.finditer(pattern_inline, updated_text))
        if not (matches_toplevel or matches_inline):
            missing.append(name)
            continue

        def _replace_value(match: re.Match[str]) -> str:
            old_value = match.group("value").strip()
            new_value = format_authoritative_parameter_value(value)
            return match.group(0).replace(old_value, new_value, 1)

        if matches_toplevel:
            updated_text = re.sub(pattern_toplevel, _replace_value, updated_text, flags=re.MULTILINE)
        if matches_inline:
            updated_text = re.sub(pattern_inline, _replace_value, updated_text)

    return updated_text, missing, update_errors
