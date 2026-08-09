from __future__ import annotations

from collections.abc import Sequence

from kindred.core.simulator.parameter_algebra_spec import classify_parameter_algebra_declaration

__all__ = [
    "is_algebra_line",
    "upsert_lines_into_algebra_section",
]

def is_algebra_line(line: str) -> bool:
    return classify_parameter_algebra_declaration(line).kind in {
        "param",
        "let",
        "invalid_step_key_identifier",
        "unsupported_bare_assignment",
    }


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
