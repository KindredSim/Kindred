from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_PAYLOAD_FIELDS = frozenset({"reactions_text"})


def normalize_source_for_analysis(text: str) -> str:
    """Return the parser's comparison form without changing authored text.

    The parser deliberately normalizes line endings, line-end whitespace, and
    outer blank space for scientific analysis.  This helper is only a pure
    comparison projection; callers must retain the original source string.
    """

    if type(text) is not str:
        raise TypeError("source text must be an exact str.")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def source_matches_analysis(source: object, analysis: object) -> bool:
    """Return whether exact authored source corresponds to retained analysis."""

    if not isinstance(source, MechanismAuthoringSource):
        return False
    normalized = getattr(analysis, "normalized_text", None)
    if type(normalized) is not str:
        return False
    return normalize_source_for_analysis(source.reactions_text) == normalized


@dataclass(frozen=True)
class MechanismAuthoringSource:
    """Complete current authored mechanism source."""

    reactions_text: str = ""

    def __post_init__(self) -> None:
        if type(self.reactions_text) is not str:
            raise TypeError("reactions_text must be an exact str.")

    @classmethod
    def from_parts(
        cls,
        *,
        reactions_text: str = "",
    ) -> "MechanismAuthoringSource":
        return cls(reactions_text=reactions_text)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "MechanismAuthoringSource":
        if not isinstance(payload, Mapping):
            raise TypeError("mechanism source payload must be a mapping.")
        missing_fields = sorted(_PAYLOAD_FIELDS - set(payload.keys()))
        if missing_fields:
            raise ValueError(
                "mechanism source payload is missing required field(s): "
                + ", ".join(missing_fields)
            )
        unknown_fields = sorted(set(payload.keys()) - _PAYLOAD_FIELDS)
        if unknown_fields:
            raise ValueError(
                "mechanism source payload has unknown field(s): "
                + ", ".join(unknown_fields)
            )
        return cls.from_parts(reactions_text=payload["reactions_text"])

    def to_payload(self) -> dict[str, str]:
        return {"reactions_text": self.reactions_text}
