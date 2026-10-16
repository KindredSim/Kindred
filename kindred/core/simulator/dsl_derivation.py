"""Small source insertion primitive for admitted DSL derivations.

The line parser and :func:`prepare_dsl_request_set` remain the sole DSL
admission boundary.  Typed coordinate rewrites are owned by the parser; this
module only inserts newly admitted scalar declarations for exact correction.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = ["insert_scalar_source_declarations"]

# The parser owns semantic admission; insertion only needs the identifier
# guard for newly materialized scalar declarations.


def insert_scalar_source_declarations(
    source_text: str,
    corrections: "Mapping[str, str]",
) -> str:
    """Insert one deterministic scalar block without rewriting authored bytes."""

    # The single caller passes exact correction output.
    if not corrections:
        return source_text
    newline_match = re.search(r"\r\n|\r|\n", source_text)
    ending = newline_match.group(0) if newline_match is not None else "\n"
    existing = {
        str(match.group("name"))
        for line in source_text.splitlines()
        for match in [
            re.match(r"^\s*(?i:param)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        ]
        if match is not None
    }
    duplicates = tuple(sorted(set(corrections).intersection(existing)))
    if duplicates:
        raise ValueError(
            "Scalar correction declaration already exists: "
            + ", ".join(duplicates)
        )
    had_final_ending = source_text.endswith(("\n", "\r"))
    prefix = source_text if not source_text or had_final_ending else source_text + ending
    declarations = ending.join(
        f"param {name} = {expression}" for name, expression in corrections.items()
    )
    return prefix + declarations + (ending if had_final_ending else "")
