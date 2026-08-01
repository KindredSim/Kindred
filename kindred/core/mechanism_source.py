from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_STATE_NETWORK_SECTION_HEADER = "# State Network"
_PAYLOAD_FIELDS = frozenset({"reactions_text", "state_network_dsl"})


def _line_starts_state_network_record(line: str) -> bool:
    lowered = str(line or "").strip().lower()
    return (
        lowered.startswith("state:")
        or lowered.startswith("edge:")
        or lowered.startswith("energy=")
    )


def _looks_like_state_network_section(lines: list[str]) -> bool:
    for line in lines:
        stripped = str(line or "").strip()
        if not stripped or stripped.startswith("#"):
            continue
        return _line_starts_state_network_record(stripped)
    return False


@dataclass(frozen=True)
class MechanismAuthoringSource:
    """Complete authored mechanism source: Reactions text plus State Network DSL."""

    reactions_text: str = ""
    state_network_dsl: str = ""

    def __post_init__(self) -> None:
        if type(self.reactions_text) is not str:
            raise TypeError("reactions_text must be an exact str.")
        if type(self.state_network_dsl) is not str:
            raise TypeError("state_network_dsl must be an exact str.")

    @classmethod
    def from_parts(cls, *, reactions_text: str = "", state_network_dsl: str = "") -> "MechanismAuthoringSource":
        return cls(
            reactions_text=str(reactions_text or ""),
            state_network_dsl=str(state_network_dsl or ""),
        )

    @classmethod
    def from_full_dsl_text(cls, full_dsl_text: str) -> "MechanismAuthoringSource":
        lines = str(full_dsl_text or "").splitlines()
        section_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().lower() == _STATE_NETWORK_SECTION_HEADER.lower()
                and _looks_like_state_network_section(lines[index + 1 :])
            ),
            None,
        )
        if section_index is None:
            return cls.from_parts(reactions_text=str(full_dsl_text or ""), state_network_dsl="")
        return cls.from_parts(
            reactions_text="\n".join(lines[:section_index]).rstrip("\n"),
            state_network_dsl="\n".join(lines[section_index + 1 :]).strip("\n"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MechanismAuthoringSource":
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
        for field in sorted(_PAYLOAD_FIELDS):
            if not isinstance(payload[field], str):
                raise TypeError(f"mechanism source payload field {field!r} must be a str.")
        return cls.from_parts(
            reactions_text=payload["reactions_text"],
            state_network_dsl=payload["state_network_dsl"],
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "reactions_text": str(self.reactions_text or ""),
            "state_network_dsl": str(self.state_network_dsl or ""),
        }

    def with_reactions_text(self, reactions_text: str) -> "MechanismAuthoringSource":
        return type(self).from_parts(
            reactions_text=str(reactions_text or ""),
            state_network_dsl=self.state_network_dsl,
        )

    def with_state_network_dsl(self, state_network_dsl: str) -> "MechanismAuthoringSource":
        return type(self).from_parts(
            reactions_text=self.reactions_text,
            state_network_dsl=str(state_network_dsl or ""),
        )

    @property
    def full_dsl(self) -> str:
        reactions_text = str(self.reactions_text or "")
        state_network_dsl = str(self.state_network_dsl or "")
        if state_network_dsl.strip():
            if reactions_text.strip():
                return f"{reactions_text}\n\n{_STATE_NETWORK_SECTION_HEADER}\n{state_network_dsl}"
            return f"{_STATE_NETWORK_SECTION_HEADER}\n{state_network_dsl}"
        return reactions_text
