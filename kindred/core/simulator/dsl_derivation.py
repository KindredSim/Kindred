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
    "coordinate_source_parts",
    "coordinate_token_matches",
    "origin_line_positions",
    "rewrite_coordinate_source_line",
    "rewrite_scalar_source_line",
    "scalar_origin",
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


def rewrite_coordinate_source_line(
    source_line: str,
    *,
    token: str,
    value: float,
) -> str:
    """Replace one admitted coordinate value without normalizing its line.

    The coordinate rewrite is intentionally local: key spelling, separators,
    comments, and all whitespace outside the numeric value remain authored
    source.  The parser has already established that the target coordinate is
    writable, so a missing or duplicate target is a fail-closed error here.
    """

    if type(source_line) is not str:
        raise TypeError("source_line must be an exact str.")
    before_comment, marker, comment = source_line.partition("#")
    matches = [
        match
        for match in re.finditer(
            r"(?P<key>[^=;,#]+)=(?P<value>[^;,#]*)",
            before_comment,
        )
        if coordinate_token_matches(match.group("key"), token)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Source does not contain exactly one authored coordinate {token!r}."
        )
    match = matches[0]
    raw_value = match.group("value")
    left_space = raw_value[: len(raw_value) - len(raw_value.lstrip())]
    right_space = raw_value[len(raw_value.rstrip()) :]
    replacement = left_space + repr(float(value)) + right_space
    return (
        before_comment[: match.start("value")]
        + replacement
        + before_comment[match.end("value") :]
        + (f"{marker}{comment}" if marker else "")
    )


def rewrite_scalar_source_line(
    source_line: str,
    *,
    name: str,
    expression: str,
) -> str:
    """Replace one scalar ``param`` RHS while retaining authored layout."""

    if type(source_line) is not str:
        raise TypeError("source_line must be an exact str.")
    if type(expression) is not str or not expression.strip():
        raise TypeError("expression must be nonblank exact text.")
    if "\n" in expression or "\r" in expression:
        raise ValueError("expression must be single-line text.")
    before_comment, marker, comment = source_line.partition("#")
    match = re.match(
        rf"^(?P<prefix>\s*(?i:param)\s+{re.escape(name)}\s*=\s*)"
        rf"(?P<rhs>.*)$",
        before_comment,
    )
    if match is None:
        raise ValueError(f"Scalar parameter {name!r} origin is invalid.")
    rhs = match.group("rhs")
    if not rhs.strip():
        raise ValueError(f"Scalar parameter {name!r} origin is invalid.")
    leading = rhs[: len(rhs) - len(rhs.lstrip())]
    trailing = rhs[len(rhs.rstrip()) :]
    return (
        match.group("prefix")
        + leading
        + expression
        + trailing
        + (f"{marker}{comment}" if marker else "")
    )


def origin_line_positions(
    analysis: "DslAnalysis",
    source_text: str,
) -> dict[int, int]:
    """Resolve immutable origin ordinals onto the exact source line indexes."""

    if type(source_text) is not str:
        raise TypeError("source_text must be an exact str.")
    raw_lines = source_text.splitlines(keepends=True)
    lines = [line.rstrip("\r\n") for line in raw_lines]
    positions: dict[int, int] = {}
    cursor = 0
    for origin in analysis.origins:
        position = next(
            (
                index
                for index in range(cursor, len(lines))
                if lines[index].rstrip() == origin.source_text.rstrip()
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

    authored_steps = [
        origin
        for origin in analysis.origins
        if any(
            arrow in origin.source_text.split("#", 1)[0]
            for arrow in ("->", "=>", "<=>")
        )
    ]
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
        rf"^\s*(?i:param)\s+{re.escape(name)}\s*=",
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
