from __future__ import annotations

from collections.abc import Sequence
import re

__all__ = [
    "extract_algebra_section_text",
    "is_algebra_line",
    "is_bare_assignment_algebra_line",
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
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_ARROW_RE = re.compile(r"<->|<=>|->|=>")
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


def _code_without_inline_comment(line: str) -> str:
    return str(line or "").split("#", 1)[0].rstrip()


def is_let_algebra_line(line: str) -> bool:
    match = _LET_LINE_RE.match(_code_without_inline_comment(line))
    if match is None:
        return False
    return not str(match.group(2) or "").lstrip().startswith("{")


def is_param_algebra_line(line: str) -> bool:
    match = _PARAM_LINE_RE.match(_code_without_inline_comment(line))
    if match is None:
        return False
    return not str(match.group(2) or "").lstrip().startswith("{")


def is_bare_assignment_algebra_line(line: str) -> bool:
    code = _code_without_inline_comment(line).strip()
    if not code or _ARROW_RE.search(code):
        return False
    if re.match(r"^(let|param)\b", code, flags=re.IGNORECASE):
        return False
    match = _ASSIGN_RE.match(code)
    if match is None:
        return False
    return str(match.group(1) or "").lower() not in _NON_ALGEBRA_ASSIGNMENT_PREFIXES


def is_algebra_line(line: str) -> bool:
    return (
        is_let_algebra_line(line)
        or is_param_algebra_line(line)
        or is_bare_assignment_algebra_line(line)
    )


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
