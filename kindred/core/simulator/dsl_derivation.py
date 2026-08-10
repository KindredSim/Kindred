"""Typed source-rewrite primitives for admitted DSL derivations.

The line parser and :func:`prepare_dsl_request_set` remain the sole DSL
admission boundary.  This module owns the small, deterministic source
coordinate rewrite stage used when a retained typed DSL request is derived
for Fit or another explicitly admitted consumer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dsl_parse import DslAnalysis, DslSourceOrigin

__all__ = [
    "authored_step_origin",
    "coordinate_source_line",
    "coordinate_source_parts",
    "coordinate_token_matches",
    "origin_line_positions",
    "scalar_origin",
    "set_coordinate_source_token",
]


_SOURCE_COORDINATE_ALIASES = {
    "k": "k",
    "kf": "kf",
    "kr": "kr",
    "keq": "keq",
    "degeneracy": "degeneracy",
    "κ": "κ",
    "kappa": "κ",
    "c0": "c0",
    "c°": "c0",
    "dg_act": "dg_act",
    "δg‡": "dg_act",
    "δg^‡": "dg_act",
    "ea": "ea",
    "dg_eq": "dg_eq",
    "δg°": "dg_eq",
    "a": "a",
}


def _canonical_source_coordinate_name(name: str) -> str:
    normalized = str(name).strip().casefold()
    return _SOURCE_COORDINATE_ALIASES.get(normalized, normalized)


def coordinate_source_parts(
    source_line: str,
) -> tuple[str, list[list[str]], str]:
    """Split one retained source line into rewriteable coordinate tokens."""

    before_comment, marker, comment = source_line.partition("#")
    prefix, separator, parameter_text = before_comment.partition(";")
    tokens: list[list[str]] = []
    if separator:
        for raw_token in re.split(r"[;,]", parameter_text):
            token = raw_token.strip()
            if not token:
                continue
            key, equals, value = token.partition("=")
            tokens.append([key.strip(), value.strip() if equals else ""])
    return prefix.rstrip(), tokens, f"{marker}{comment}" if marker else ""


def coordinate_source_line(
    prefix: str,
    tokens: list[list[str]],
    comment: str,
) -> str:
    """Render one source line after a typed coordinate rewrite."""

    body = prefix.rstrip()
    if tokens:
        body += " ; " + ", ".join(
            f"{key}={value}" if value else f"{key}="
            for key, value in tokens
        )
    if comment:
        body += f" {comment.strip()}"
    return body


def coordinate_token_matches(candidate: str, target: str) -> bool:
    """Return whether a source token denotes one canonical coordinate."""

    candidate_name = str(candidate).strip()
    target_name = str(target).strip()
    # ``K`` is intentionally not a Keq alias.  Keep this rejection before
    # Unicode/casefold normalization so an unadvertised legacy spelling
    # cannot become writable through the rewrite stage.
    if candidate_name == "K":
        return False
    return _canonical_source_coordinate_name(candidate_name) == (
        _canonical_source_coordinate_name(target_name)
    )


def set_coordinate_source_token(
    tokens: list[list[str]],
    *,
    token: str,
    value: float,
) -> None:
    """Replace one coordinate token and remove duplicate authored spellings."""

    # ``repr`` is Python's shortest decimal representation that round-trips
    # the binary64 value exactly.  A fixed significant-digit format silently
    # changes slider coordinates during source publication.
    rendered = repr(float(value))
    matches = [
        index
        for index, pair in enumerate(tokens)
        if coordinate_token_matches(pair[0], token)
    ]
    if matches:
        target = matches[0]
        tokens[target] = [token, rendered]
        for duplicate in reversed(matches[1:]):
            tokens.pop(duplicate)
        return
    tokens.append([token, rendered])


def origin_line_positions(
    analysis: "DslAnalysis",
) -> dict[int, int]:
    """Resolve immutable origin ordinals onto canonical source line indexes."""

    lines = analysis.normalized_text.splitlines()
    positions: dict[int, int] = {}
    cursor = 0
    for origin in analysis.origins:
        position = next(
            (
                index
                for index in range(cursor, len(lines))
                if lines[index] == origin.source_text
            ),
            None,
        )
        if position is None:
            raise ValueError(
                "Prepared DSL origin no longer matches canonical source."
            )
        positions[origin.authored_ordinal] = position
        cursor = position + 1
    return positions


def authored_step_origin(
    analysis: "DslAnalysis",
    *,
    step_index: int,
) -> "DslSourceOrigin":
    """Return the exact retained source origin for one authored step."""

    authored_steps: list[DslSourceOrigin] = []
    for line in analysis.normalized_text.splitlines():
        origin = next(
            (
                item
                for item in analysis.origins
                if item.source_text == line
                and item.authored_ordinal
                not in {
                    candidate.authored_ordinal
                    for candidate in authored_steps
                }
            ),
            None,
        )
        if origin is None:
            continue
        code = line.split("#", 1)[0].strip()
        if any(arrow in code for arrow in ("->", "=>", "<=>")):
            authored_steps.append(origin)
    if not 1 <= step_index <= len(authored_steps):
        raise LookupError(
            f"Mechanism step {step_index} has no exact source origin."
        )
    return authored_steps[step_index - 1]


def scalar_origin(
    analysis: "DslAnalysis",
    *,
    name: str,
) -> "DslSourceOrigin":
    """Return the unique retained source origin for one scalar declaration."""

    pattern = re.compile(
        rf"^\s*param\s+{re.escape(name)}\s*=",
    )
    matches = [
        origin
        for origin in analysis.origins
        if pattern.match(origin.source_text.split("#", 1)[0])
    ]
    if len(matches) != 1:
        raise LookupError(
            f"Scalar parameter {name!r} has no unique source origin."
        )
    return matches[0]
