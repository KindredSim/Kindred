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
    additions = [
        str(x).rstrip("\r\n")
        for x in (lines_to_add or [])
        if str(x).strip()
    ]
    if not additions:
        return str(dsl_text or "")

    raw_lines = str(dsl_text or "").splitlines(keepends=True)
    lines = [line.rstrip("\r\n") for line in raw_lines]
    endings = [
        line[len(content):]
        for line, content in zip(raw_lines, lines)
    ]
    if not raw_lines and dsl_text:
        lines = [str(dsl_text)]
        endings = [""]
    new_lines = list(lines)
    insert_at = len(lines)
    for idx, raw in enumerate(lines):
        if is_algebra_line(raw):
            insert_at = idx + 1

    new_lines[insert_at:insert_at] = additions
    default_ending = next(
        (ending for ending in endings if ending),
        "\n",
    )
    had_final_newline = bool(endings and endings[-1])
    new_endings = list(endings)
    new_endings[insert_at:insert_at] = [default_ending] * len(additions)
    if insert_at == len(endings) and endings and not endings[-1]:
        new_endings[insert_at - 1] = default_ending
    if len(new_endings) < len(new_lines):
        new_endings.extend([default_ending] * (len(new_lines) - len(new_endings)))
    if new_endings:
        new_endings[-1] = default_ending if had_final_newline else ""
    return "".join(
        line + ending
        for line, ending in zip(new_lines, new_endings)
    )
