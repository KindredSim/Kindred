from __future__ import annotations

from collections.abc import Sequence
import re

from kindred.core.simulator.parameter_algebra_spec import classify_parameter_algebra_declaration

__all__ = [
    "extract_algebra_section_text",
    "is_algebra_line",
    "is_let_algebra_line",
    "is_param_algebra_line",
    "upsert_lines_into_algebra_section",
]

_LET_LINE_RE = re.compile(
    r"^\s*let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)
_PARAM_LINE_RE = re.compile(
    r"^\s*param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)

def _code_without_inline_comment(line: str) -> str:
    return str(line or "").split("#", 1)[0].rstrip()


def is_let_algebra_line(line: str) -> bool:
    classification = classify_parameter_algebra_declaration(line)
    if classification.kind == "let":
        return True
    return bool(
        classification.kind == "invalid_step_key_identifier"
        and _LET_LINE_RE.match(_code_without_inline_comment(line))
    )


def is_param_algebra_line(line: str) -> bool:
    classification = classify_parameter_algebra_declaration(line)
    if classification.kind == "param":
        return True
    return bool(
        classification.kind == "invalid_step_key_identifier"
        and _PARAM_LINE_RE.match(_code_without_inline_comment(line))
    )


def is_algebra_line(line: str) -> bool:
    return classify_parameter_algebra_declaration(line).kind in {
        "param",
        "let",
        "invalid_step_key_identifier",
        "unsupported_bare_assignment",
    }


def extract_algebra_section_text(dsl_text: str) -> str:
    """
    Return algebra lines found anywhere in the DSL text.
    """
    out: list[str] = []
    for raw in str(dsl_text or "").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if is_algebra_line(raw):
            out.append(raw.rstrip("\n"))
    return "\n".join(out).rstrip("\n")


def upsert_lines_into_algebra_section(
    dsl_text: str,
    lines_to_add: Sequence[str],
) -> str:
    """
    Insert algebra lines after the last existing algebra line, or append at end.
    """
    additions = [str(x).rstrip("\n") for x in (lines_to_add or []) if str(x).strip()]
    if not additions:
        return str(dsl_text or "")

    lines = str(dsl_text or "").splitlines()
    new_lines = list(lines)
    insert_at = len(lines)
    for idx, raw in enumerate(lines):
        if is_algebra_line(raw):
            insert_at = idx + 1

    new_lines[insert_at:insert_at] = additions
    return "\n".join(new_lines).rstrip("\n") + "\n"
