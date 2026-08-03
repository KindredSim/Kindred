from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_PAYLOAD_FIELDS = frozenset({"reactions_text"})


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

    def with_reactions_text(
        self,
        reactions_text: str,
    ) -> "MechanismAuthoringSource":
        return type(self).from_parts(reactions_text=reactions_text)
