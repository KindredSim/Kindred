from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_PAYLOAD_FIELDS = frozenset({"reactions_text"})
INITIAL_PROVENANCE_PREFIX = "# Kindred IC provenance | "


def restore_initial_condition_declarations(text: str) -> str:
    """Restore recorded IC source for reuse, preserving all other text."""
    return "".join(
        line.removeprefix(INITIAL_PROVENANCE_PREFIX)
        for line in text.splitlines(keepends=True)
    )


@dataclass(frozen=True)
class MechanismAuthoringSource:
    """Complete current authored mechanism source."""

    reactions_text: str = ""

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
        """Admit an archive's decoded mechanism-source object."""
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
        if type(payload["reactions_text"]) is not str:
            raise TypeError("reactions_text must be an exact str.")
        return cls.from_parts(reactions_text=payload["reactions_text"])

    def to_payload(self) -> dict[str, str]:
        return {"reactions_text": self.reactions_text}
